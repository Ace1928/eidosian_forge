import contextlib
import logging
import logging.handlers
import os
import re
import subprocess
import sys
import tempfile
from humanfriendly.compat import StringIO
from humanfriendly.terminal import ANSI_COLOR_CODES, ANSI_CSI, ansi_style, ansi_wrap
from humanfriendly.testing import PatchedAttribute, PatchedItem, TestCase, retry
from humanfriendly.text import format, random_string
import coloredlogs
import coloredlogs.cli
from coloredlogs import (
from coloredlogs.demo import demonstrate_colored_logging
from coloredlogs.syslog import SystemLogging, is_syslog_supported, match_syslog_handler
from coloredlogs.converter import (
from capturer import CaptureOutput
from verboselogs import VerboseLogger
class ColoredLogsTestCase(TestCase):
    """Container for the `coloredlogs` tests."""

    def find_system_log(self):
        """Find the system log file or skip the current test."""
        filename = '/var/log/system.log' if sys.platform == 'darwin' else '/var/log/syslog' if 'linux' in sys.platform else None
        if not filename:
            self.skipTest('Location of system log file unknown!')
        elif not os.path.isfile(filename):
            self.skipTest('System log file not found! (%s)' % filename)
        elif not os.access(filename, os.R_OK):
            self.skipTest('Insufficient permissions to read system log file! (%s)' % filename)
        else:
            return filename

    def test_level_to_number(self):
        """Make sure :func:`level_to_number()` works as intended."""
        assert level_to_number('debug') == logging.DEBUG
        assert level_to_number('info') == logging.INFO
        assert level_to_number('warning') == logging.WARNING
        assert level_to_number('error') == logging.ERROR
        assert level_to_number('fatal') == logging.FATAL
        assert level_to_number('bogus-level') == logging.INFO

    def test_find_hostname(self):
        """Make sure :func:`~find_hostname()` works correctly."""
        assert find_hostname()
        fd, temporary_file = tempfile.mkstemp()
        try:
            with open(temporary_file, 'w') as handle:
                handle.write('first line\n')
                handle.write('second line\n')
            CHROOT_FILES.insert(0, temporary_file)
            assert find_hostname() == 'first line'
        finally:
            CHROOT_FILES.pop(0)
            os.unlink(temporary_file)
        try:
            CHROOT_FILES.insert(0, temporary_file)
            assert find_hostname()
        finally:
            CHROOT_FILES.pop(0)

    def test_host_name_filter(self):
        """Make sure :func:`install()` integrates with :class:`~coloredlogs.HostNameFilter()`."""
        install(fmt='%(hostname)s')
        with CaptureOutput() as capturer:
            logging.info('A truly insignificant message ..')
            output = capturer.get_text()
            assert find_hostname() in output

    def test_program_name_filter(self):
        """Make sure :func:`install()` integrates with :class:`~coloredlogs.ProgramNameFilter()`."""
        install(fmt='%(programname)s')
        with CaptureOutput() as capturer:
            logging.info('A truly insignificant message ..')
            output = capturer.get_text()
            assert find_program_name() in output

    def test_username_filter(self):
        """Make sure :func:`install()` integrates with :class:`~coloredlogs.UserNameFilter()`."""
        install(fmt='%(username)s')
        with CaptureOutput() as capturer:
            logging.info('A truly insignificant message ..')
            output = capturer.get_text()
            assert find_username() in output

    def test_system_logging(self):
        """Make sure the :class:`coloredlogs.syslog.SystemLogging` context manager works."""
        system_log_file = self.find_system_log()
        expected_message = random_string(50)
        with SystemLogging(programname='coloredlogs-test-suite') as syslog:
            if not syslog:
                return self.skipTest("couldn't connect to syslog daemon")
            logging.error('%s', expected_message)
        retry(lambda: check_contents(system_log_file, expected_message, True))

    def test_system_logging_override(self):
        """Make sure the :class:`coloredlogs.syslog.is_syslog_supported` respects the override."""
        with PatchedItem(os.environ, 'COLOREDLOGS_SYSLOG', 'true'):
            assert is_syslog_supported() is True
        with PatchedItem(os.environ, 'COLOREDLOGS_SYSLOG', 'false'):
            assert is_syslog_supported() is False

    def test_syslog_shortcut_simple(self):
        """Make sure that ``coloredlogs.install(syslog=True)`` works."""
        system_log_file = self.find_system_log()
        expected_message = random_string(50)
        with cleanup_handlers():
            coloredlogs.install(syslog=True)
            logging.error('%s', expected_message)
        retry(lambda: check_contents(system_log_file, expected_message, True))

    def test_syslog_shortcut_enhanced(self):
        """Make sure that ``coloredlogs.install(syslog='warning')`` works."""
        system_log_file = self.find_system_log()
        the_expected_message = random_string(50)
        not_an_expected_message = random_string(50)
        with cleanup_handlers():
            coloredlogs.install(syslog='error')
            logging.warning('%s', not_an_expected_message)
            logging.error('%s', the_expected_message)
        retry(lambda: check_contents(system_log_file, the_expected_message, True))
        retry(lambda: check_contents(system_log_file, not_an_expected_message, False))

    def test_name_normalization(self):
        """Make sure :class:`~coloredlogs.NameNormalizer` works as intended."""
        nn = NameNormalizer()
        for canonical_name in ['debug', 'info', 'warning', 'error', 'critical']:
            assert nn.normalize_name(canonical_name) == canonical_name
            assert nn.normalize_name(canonical_name.upper()) == canonical_name
        assert nn.normalize_name('warn') == 'warning'
        assert nn.normalize_name('fatal') == 'critical'

    def test_style_parsing(self):
        """Make sure :func:`~coloredlogs.parse_encoded_styles()` works as intended."""
        encoded_styles = 'debug=green;warning=yellow;error=red;critical=red,bold'
        decoded_styles = parse_encoded_styles(encoded_styles, normalize_key=lambda k: k.upper())
        assert sorted(decoded_styles.keys()) == sorted(['debug', 'warning', 'error', 'critical'])
        assert decoded_styles['debug']['color'] == 'green'
        assert decoded_styles['warning']['color'] == 'yellow'
        assert decoded_styles['error']['color'] == 'red'
        assert decoded_styles['critical']['color'] == 'red'
        assert decoded_styles['critical']['bold'] is True

    def test_is_verbose(self):
        """Make sure is_verbose() does what it should :-)."""
        set_level(logging.INFO)
        assert not is_verbose()
        set_level(logging.DEBUG)
        assert is_verbose()
        set_level(logging.VERBOSE)
        assert is_verbose()

    def test_increase_verbosity(self):
        """Make sure increase_verbosity() respects default and custom levels."""
        set_level(logging.INFO)
        assert get_level() == logging.INFO
        increase_verbosity()
        assert get_level() == logging.VERBOSE
        increase_verbosity()
        assert get_level() == logging.DEBUG
        increase_verbosity()
        assert get_level() == logging.SPAM
        increase_verbosity()
        assert get_level() == logging.NOTSET
        increase_verbosity()
        assert get_level() == logging.NOTSET

    def test_decrease_verbosity(self):
        """Make sure decrease_verbosity() respects default and custom levels."""
        set_level(logging.INFO)
        assert get_level() == logging.INFO
        decrease_verbosity()
        assert get_level() == logging.NOTICE
        decrease_verbosity()
        assert get_level() == logging.WARNING
        decrease_verbosity()
        assert get_level() == logging.SUCCESS
        decrease_verbosity()
        assert get_level() == logging.ERROR
        decrease_verbosity()
        assert get_level() == logging.CRITICAL
        decrease_verbosity()
        assert get_level() == logging.CRITICAL

    def test_level_discovery(self):
        """Make sure find_defined_levels() always reports the levels defined in Python's standard library."""
        defined_levels = find_defined_levels()
        level_values = defined_levels.values()
        for number in (0, 10, 20, 30, 40, 50):
            assert number in level_values

    def test_walk_propagation_tree(self):
        """Make sure walk_propagation_tree() properly walks the tree of loggers."""
        root, parent, child, grand_child = self.get_logger_tree()
        loggers = list(walk_propagation_tree(grand_child))
        assert loggers == [grand_child, child, parent, root]
        child.propagate = False
        loggers = list(walk_propagation_tree(grand_child))
        assert loggers == [grand_child, child]

    def test_find_handler(self):
        """Make sure find_handler() works as intended."""
        root, parent, child, grand_child = self.get_logger_tree()
        stream_handler = logging.StreamHandler()
        syslog_handler = logging.handlers.SysLogHandler()
        child.addHandler(stream_handler)
        parent.addHandler(syslog_handler)
        matched_handler, matched_logger = find_handler(grand_child, lambda h: isinstance(h, logging.Handler))
        assert matched_handler is stream_handler
        matched_handler, matched_logger = find_handler(child, lambda h: isinstance(h, logging.handlers.SysLogHandler))
        assert matched_handler is syslog_handler

    def get_logger_tree(self):
        """Create and return a tree of loggers."""
        root = logging.getLogger()
        parent_name = random_string()
        parent = logging.getLogger(parent_name)
        child_name = '%s.%s' % (parent_name, random_string())
        child = logging.getLogger(child_name)
        grand_child_name = '%s.%s' % (child_name, random_string())
        grand_child = logging.getLogger(grand_child_name)
        return (root, parent, child, grand_child)

    def test_support_for_milliseconds(self):
        """Make sure milliseconds are hidden by default but can be easily enabled."""
        stream = StringIO()
        install(reconfigure=True, stream=stream)
        logging.info('This should not include milliseconds.')
        assert all(map(PLAIN_TEXT_PATTERN.match, stream.getvalue().splitlines()))
        stream = StringIO()
        install(milliseconds=True, reconfigure=True, stream=stream)
        logging.info('This should include milliseconds.')
        assert all(map(PATTERN_INCLUDING_MILLISECONDS.match, stream.getvalue().splitlines()))

    def test_support_for_milliseconds_directive(self):
        """Make sure milliseconds using the ``%f`` directive are supported."""
        stream = StringIO()
        install(reconfigure=True, stream=stream, datefmt='%Y-%m-%dT%H:%M:%S.%f%z')
        logging.info('This should be timestamped according to #45.')
        assert re.match('^\\d{4}-\\d{2}-\\d{2}T\\d{2}:\\d{2}:\\d{2}\\.\\d{3}[+-]\\d{4}\\s', stream.getvalue())

    def test_plain_text_output_format(self):
        """Inspect the plain text output of coloredlogs."""
        logger = VerboseLogger(random_string(25))
        stream = StringIO()
        install(level=logging.NOTSET, logger=logger, stream=stream)
        logger.setLevel(logging.INFO)
        logger.debug('No one should see this message.')
        assert len(stream.getvalue().strip()) == 0
        logger.setLevel(logging.NOTSET)
        for method, severity in ((logger.debug, 'DEBUG'), (logger.info, 'INFO'), (logger.verbose, 'VERBOSE'), (logger.warning, 'WARNING'), (logger.error, 'ERROR'), (logger.critical, 'CRITICAL')):
            try:
                logger._cache.clear()
            except AttributeError:
                pass
            text = 'This is a message with severity %r.' % severity.lower()
            method(text)
            output = stream.getvalue()
            lines = output.splitlines()
            last_line = lines[-1]
            assert text in last_line
            assert severity in last_line
            assert PLAIN_TEXT_PATTERN.match(last_line)

    def test_dynamic_stderr_lookup(self):
        """Make sure coloredlogs.install() uses StandardErrorHandler when possible."""
        coloredlogs.install()
        initial_stream = StringIO()
        initial_text = 'Which stream will receive this text?'
        with PatchedAttribute(sys, 'stderr', initial_stream):
            logging.info(initial_text)
        assert initial_text in initial_stream.getvalue()
        subsequent_stream = StringIO()
        subsequent_text = 'And which stream will receive this other text?'
        with PatchedAttribute(sys, 'stderr', subsequent_stream):
            logging.info(subsequent_text)
        assert subsequent_text in subsequent_stream.getvalue()

    def test_force_enable(self):
        """Make sure ANSI escape sequences can be forced (bypassing auto-detection)."""
        interpreter = subprocess.Popen([sys.executable, '-c', ';'.join(['import coloredlogs, logging', 'coloredlogs.install(isatty=True)', "logging.info('Hello world')"])], stderr=subprocess.PIPE)
        stdout, stderr = interpreter.communicate()
        assert ANSI_CSI in stderr.decode('UTF-8')

    def test_auto_disable(self):
        """
        Make sure ANSI escape sequences are not emitted when logging output is being redirected.

        This is a regression test for https://github.com/xolox/python-coloredlogs/issues/100.

        It works as follows:

        1. We mock an interactive terminal using 'capturer' to ensure that this
           test works inside test drivers that capture output (like pytest).

        2. We launch a subprocess (to ensure a clean process state) where
           stderr is captured but stdout is not, emulating issue #100.

        3. The output captured on stderr contained ANSI escape sequences after
           this test was written and before the issue was fixed, so now this
           serves as a regression test for issue #100.
        """
        with CaptureOutput():
            interpreter = subprocess.Popen([sys.executable, '-c', ';'.join(['import coloredlogs, logging', 'coloredlogs.install()', "logging.info('Hello world')"])], stderr=subprocess.PIPE)
            stdout, stderr = interpreter.communicate()
            assert ANSI_CSI not in stderr.decode('UTF-8')

    def test_env_disable(self):
        """Make sure ANSI escape sequences can be disabled using ``$NO_COLOR``."""
        with PatchedItem(os.environ, 'NO_COLOR', 'I like monochrome'):
            with CaptureOutput() as capturer:
                subprocess.check_call([sys.executable, '-c', ';'.join(['import coloredlogs, logging', 'coloredlogs.install()', "logging.info('Hello world')"])])
                output = capturer.get_text()
                assert ANSI_CSI not in output

    def test_html_conversion(self):
        """Check the conversion from ANSI escape sequences to HTML."""
        for color_name, ansi_code in ANSI_COLOR_CODES.items():
            ansi_encoded_text = 'plain text followed by %s text' % ansi_wrap(color_name, color=color_name)
            expected_html = format('<code>plain text followed by <span style="color:{css}">{name}</span> text</code>', css=EIGHT_COLOR_PALETTE[ansi_code], name=color_name)
            self.assertEqual(expected_html, convert(ansi_encoded_text))
        expected_html = '<code><span style="color:#FF0">bright yellow</span></code>'
        self.assertEqual(expected_html, convert(ansi_wrap('bright yellow', color='yellow', bright=True)))
        expected_html = '<code><span style="background-color:#DE382B">red background</span></code>'
        self.assertEqual(expected_html, convert(ansi_wrap('red background', background='red')))
        expected_html = '<code><span style="background-color:#F00">bright red background</span></code>'
        self.assertEqual(expected_html, convert(ansi_wrap('bright red background', background='red', bright=True)))
        expected_html = '<code><span style="color:#FFAF00">256 color mode foreground</span></code>'
        self.assertEqual(expected_html, convert(ansi_wrap('256 color mode foreground', color=214)))
        expected_html = '<code><span style="background-color:#AF0000">256 color mode background</span></code>'
        self.assertEqual(expected_html, convert(ansi_wrap('256 color mode background', background=124)))
        expected_html = '<code>plain text expected</code>'
        self.assertEqual(expected_html, convert('\x1b[38;5;256mplain text expected\x1b[0m'))
        expected_html = '<code><span style="font-weight:bold">bold text</span></code>'
        self.assertEqual(expected_html, convert(ansi_wrap('bold text', bold=True)))
        expected_html = '<code><span style="text-decoration:underline">underlined text</span></code>'
        self.assertEqual(expected_html, convert(ansi_wrap('underlined text', underline=True)))
        expected_html = '<code><span style="text-decoration:line-through">strike-through text</span></code>'
        self.assertEqual(expected_html, convert(ansi_wrap('strike-through text', strike_through=True)))
        expected_html = '<code><span style="background-color:#FFC706;color:#000">inverse</span></code>'
        self.assertEqual(expected_html, convert(ansi_wrap('inverse', color='yellow', inverse=True)))
        for sample_text in ('www.python.org', 'http://coloredlogs.rtfd.org', 'https://coloredlogs.rtfd.org'):
            sample_url = sample_text if '://' in sample_text else 'http://' + sample_text
            expected_html = '<code><a href="%s" style="color:inherit">%s</a></code>' % (sample_url, sample_text)
            self.assertEqual(expected_html, convert(sample_text))
        reset_short_hand = '\x1b[0m'
        blue_underlined = ansi_style(color='blue', underline=True)
        ansi_encoded_text = '<%shttps://coloredlogs.readthedocs.io%s>' % (blue_underlined, reset_short_hand)
        expected_html = '<code>&lt;<span style="color:#006FB8;text-decoration:underline"><a href="https://coloredlogs.readthedocs.io" style="color:inherit">https://coloredlogs.readthedocs.io</a></span>&gt;</code>'
        self.assertEqual(expected_html, convert(ansi_encoded_text))

    def test_output_interception(self):
        """Test capturing of output from external commands."""
        expected_output = 'testing, 1, 2, 3 ..'
        actual_output = capture(['echo', expected_output])
        assert actual_output.strip() == expected_output.strip()

    def test_enable_colored_cron_mailer(self):
        """Test that automatic ANSI to HTML conversion when running under ``cron`` can be enabled."""
        with PatchedItem(os.environ, 'CONTENT_TYPE', 'text/html'):
            with ColoredCronMailer() as mailer:
                assert mailer.is_enabled

    def test_disable_colored_cron_mailer(self):
        """Test that automatic ANSI to HTML conversion when running under ``cron`` can be disabled."""
        with PatchedItem(os.environ, 'CONTENT_TYPE', 'text/plain'):
            with ColoredCronMailer() as mailer:
                assert not mailer.is_enabled

    def test_auto_install(self):
        """Test :func:`coloredlogs.auto_install()`."""
        needle = random_string()
        command_line = [sys.executable, '-c', 'import logging; logging.info(%r)' % needle]
        with CaptureOutput() as capturer:
            os.environ['COLOREDLOGS_AUTO_INSTALL'] = 'false'
            subprocess.check_call(command_line)
            output = capturer.get_text()
        assert needle not in output
        with CaptureOutput() as capturer:
            os.environ['COLOREDLOGS_AUTO_INSTALL'] = 'true'
            subprocess.check_call(command_line)
            output = capturer.get_text()
        assert needle in output

    def test_cli_demo(self):
        """Test the command line colored logging demonstration."""
        with CaptureOutput() as capturer:
            main('coloredlogs', '--demo')
            output = capturer.get_text()
        for name in ('debug', 'info', 'warning', 'error', 'critical'):
            assert name.upper() in output

    def test_cli_conversion(self):
        """Test the command line HTML conversion."""
        output = main('coloredlogs', '--convert', 'coloredlogs', '--demo', capture=True)
        assert '<span' in output

    def test_empty_conversion(self):
        """
        Test that conversion of empty output produces no HTML.

        This test was added because I found that ``coloredlogs --convert`` when
        used in a cron job could cause cron to send out what appeared to be
        empty emails. On more careful inspection the body of those emails was
        ``<code></code>``. By not emitting the wrapper element when no other
        HTML is generated, cron will not send out an email.
        """
        output = main('coloredlogs', '--convert', 'true', capture=True)
        assert not output.strip()

    def test_implicit_usage_message(self):
        """Test that the usage message is shown when no actions are given."""
        assert 'Usage:' in main('coloredlogs', capture=True)

    def test_explicit_usage_message(self):
        """Test that the usage message is shown when ``--help`` is given."""
        assert 'Usage:' in main('coloredlogs', '--help', capture=True)

    def test_custom_record_factory(self):
        """
        Test that custom LogRecord factories are supported.

        This test is a bit convoluted because the logging module suppresses
        exceptions. We monkey patch the method suspected of encountering
        exceptions so that we can tell after it was called whether any
        exceptions occurred (despite the exceptions not propagating).
        """
        if not hasattr(logging, 'getLogRecordFactory'):
            return self.skipTest('this test requires Python >= 3.2')
        exceptions = []
        original_method = ColoredFormatter.format
        original_factory = logging.getLogRecordFactory()

        def custom_factory(*args, **kwargs):
            record = original_factory(*args, **kwargs)
            record.custom_attribute = 3737844653
            return record

        def custom_method(*args, **kw):
            try:
                return original_method(*args, **kw)
            except Exception as e:
                exceptions.append(e)
                raise
        with PatchedAttribute(ColoredFormatter, 'format', custom_method):
            logging.setLogRecordFactory(custom_factory)
            try:
                demonstrate_colored_logging()
            finally:
                logging.setLogRecordFactory(original_factory)
        assert not exceptions