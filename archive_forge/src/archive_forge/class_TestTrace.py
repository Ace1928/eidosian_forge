import errno
import logging
import os
import re
import sys
import tempfile
from io import StringIO
from .. import debug, errors, trace
from ..trace import (_rollover_trace_maybe, be_quiet, get_verbosity_level,
from . import TestCase, TestCaseInTempDir, TestSkipped, features
class TestTrace(TestCase):

    def test_format_sys_exception(self):
        try:
            raise NotImplementedError('time travel')
        except NotImplementedError:
            err = _format_exception()
        self.assertContainsRe(err, '^brz: ERROR: NotImplementedError: time travel')
        self.assertContainsRe(err, 'Breezy has encountered an internal error.')

    def test_format_interrupt_exception(self):
        try:
            raise KeyboardInterrupt()
        except KeyboardInterrupt:
            msg = _format_exception()
        self.assertEqual(msg, 'brz: interrupted\n')

    def test_format_memory_error(self):
        try:
            raise MemoryError()
        except MemoryError:
            msg = _format_exception()
        self.assertEqual(msg, 'brz: out of memory\nUse -Dmem_dump to dump memory to a file.\n')

    def test_format_mem_dump(self):
        self.requireFeature(features.meliae)
        debug.debug_flags.add('mem_dump')
        try:
            raise MemoryError()
        except MemoryError:
            msg = _format_exception()
        self.assertStartsWith(msg, 'brz: out of memory\nMemory dumped to ')

    def test_format_os_error(self):
        try:
            os.rmdir('nosuchfile22222')
        except OSError as e:
            e_str = str(e)
            msg = _format_exception()
        self.assertEqual('brz: ERROR: {}\n'.format(e_str), msg)

    def test_format_io_error(self):
        try:
            open('nosuchfile22222')
        except OSError:
            msg = _format_exception()
        self.assertContainsRe(msg, '^brz: ERROR: \\[Errno .*\\] .*nosuchfile')

    def test_format_pywintypes_error(self):
        self.requireFeature(features.pywintypes)
        import pywintypes
        import win32file
        try:
            win32file.RemoveDirectory('nosuchfile22222')
        except pywintypes.error:
            msg = _format_exception()
        self.assertContainsRe(msg, "^brz: ERROR: \\(2, 'RemoveDirectory[AW]?', .*\\)")

    def test_format_sockets_error(self):
        try:
            import socket
            sock = socket.socket()
            sock.send(b'This should fail.')
        except OSError:
            msg = _format_exception()
        self.assertNotContainsRe(msg, 'Traceback \\(most recent call last\\):')

    def test_format_unicode_error(self):
        try:
            raise errors.CommandError('argument fooµ does not exist')
        except errors.CommandError:
            msg = _format_exception()
        expected = 'brz: ERROR: argument fooµ does not exist\n'
        self.assertEqual(msg, expected)

    def test_format_exception(self):
        """Short formatting of brz exceptions"""
        try:
            raise errors.NotBranchError('wibble')
        except errors.NotBranchError:
            msg = _format_exception()
        self.assertEqual(msg, 'brz: ERROR: Not a branch: "wibble".\n')

    def test_report_external_import_error(self):
        """Short friendly message for missing system modules."""
        try:
            import ImaginaryModule
        except ImportError:
            msg = _format_exception()
        else:
            self.fail('somehow succeeded in importing %r' % ImaginaryModule)
        self.assertContainsRe(msg, "^brz: ERROR: No module named '?ImaginaryModule'?\nYou may need to install this Python library separately.\n$")

    def test_report_import_syntax_error(self):
        try:
            raise ImportError('syntax error')
        except ImportError:
            msg = _format_exception()
        self.assertContainsRe(msg, 'Breezy has encountered an internal error')

    def test_trace_unicode(self):
        """Write Unicode to trace log"""
        self.log('the unicode character for benzene is ⌬')
        log = self.get_log()
        self.assertContainsRe(log, 'the unicode character for benzene is')

    def test_trace_argument_unicode(self):
        """Write a Unicode argument to the trace log"""
        mutter('the unicode character for benzene is %s', '⌬')
        log = self.get_log()
        self.assertContainsRe(log, 'the unicode character')

    def test_trace_argument_utf8(self):
        """Write a Unicode argument to the trace log"""
        mutter('the unicode character for benzene is %s', '⌬'.encode())
        log = self.get_log()
        self.assertContainsRe(log, 'the unicode character')

    def test_trace_argument_exception(self):
        err = Exception('an error')
        mutter('can format stringable classes %s', err)
        log = self.get_log()
        self.assertContainsRe(log, 'can format stringable classes an error')

    def test_report_broken_pipe(self):
        try:
            raise OSError(errno.EPIPE, 'broken pipe foofofo')
        except OSError:
            msg = _format_exception()
            self.assertEqual(msg, 'brz: broken pipe\n')
        else:
            self.fail('expected error not raised')

    def assertLogContainsLine(self, log, string):
        """Assert log contains a line including log timestamp."""
        self.assertContainsRe(log, '(?m)^\\d+\\.\\d+  ' + re.escape(string))

    def test_mutter_callsite_1(self):
        """mutter_callsite can capture 1 level of stack frame."""
        mutter_callsite(1, 'foo %s', 'a string')
        log = self.get_log()
        self.assertLogContainsLine(log, 'foo a string\nCalled from:\n')
        self.assertContainsRe(log, 'test_trace\\.py", line \\d+, in test_mutter_callsite_1\n')
        self.assertEndsWith(log, ' "a string")\n')

    def test_mutter_callsite_2(self):
        """mutter_callsite can capture 2 levels of stack frame."""
        mutter_callsite(2, 'foo %s', 'a string')
        log = self.get_log()
        self.assertLogContainsLine(log, 'foo a string\nCalled from:\n')
        self.assertContainsRe(log, 'test_trace.py", line \\d+, in test_mutter_callsite_2\n')
        self.assertEndsWith(log, ' "a string")\n')

    def test_mutter_never_fails(self):
        """Even with unencodable input mutter should not raise errors."""
        mutter('can write unicode §')
        mutter('can interpolate unicode %s', '§')
        mutter(b'can write bytes \xa7')
        mutter('can repr bytes %r', b'\xa7')
        mutter('can interpolate bytes %s', b'\xa7')
        log = self.get_log()
        self.assertContainsRe(log, ".* +can write unicode §\n.* +can interpolate unicode §\n.* +can write bytes �\n.* +can repr bytes b'\\\\xa7'\n.* +can interpolate bytes (?:�|b'\\\\xa7')\n")

    def test_show_error(self):
        show_error('error1')
        show_error('error2 µ blah')
        show_error('arg: %s', 'blah')
        show_error('arg2: %(key)s', {'key': 'stuff'})
        try:
            raise Exception('oops')
        except BaseException:
            show_error('kwarg', exc_info=True)
        log = self.get_log()
        self.assertContainsRe(log, 'error1')
        self.assertContainsRe(log, 'error2 µ blah')
        self.assertContainsRe(log, 'arg: blah')
        self.assertContainsRe(log, 'arg2: stuff')
        self.assertContainsRe(log, 'kwarg')
        self.assertContainsRe(log, 'Traceback \\(most recent call last\\):')
        self.assertContainsRe(log, 'File ".*test_trace.py", line .*, in test_show_error')
        self.assertContainsRe(log, 'raise Exception\\("oops"\\)')
        self.assertContainsRe(log, 'Exception: oops')

    def test_push_log_file(self):
        """Can push and pop log file, and this catches mutter messages.

        This is primarily for use in the test framework.
        """
        tmp1 = tempfile.NamedTemporaryFile()
        tmp2 = tempfile.NamedTemporaryFile()
        try:
            memento1 = push_log_file(tmp1)
            mutter('comment to file1')
            try:
                memento2 = push_log_file(tmp2)
                try:
                    mutter('comment to file2')
                finally:
                    pop_log_file(memento2)
                mutter('again to file1')
            finally:
                pop_log_file(memento1)
            tmp1.seek(0)
            self.assertContainsRe(tmp1.read(), b'\\d+\\.\\d+  comment to file1\n\\d+\\.\\d+  again to file1\n')
            tmp2.seek(0)
            self.assertContainsRe(tmp2.read(), b'\\d+\\.\\d+  comment to file2\n')
        finally:
            tmp1.close()
            tmp2.close()

    def test__open_brz_log_uses_stderr_for_failures(self):
        self.overrideAttr(sys, 'stderr', StringIO())
        self.overrideEnv('BRZ_LOG', '/no-such-dir/brz.log')
        self.overrideAttr(trace, '_brz_log_filename')
        logf = trace._open_brz_log()
        if os.path.isdir('/no-such-dir'):
            raise TestSkipped('directory creation succeeded')
        self.assertIs(None, logf)
        self.assertContainsRe(sys.stderr.getvalue(), "failed to open trace file: .* '/no-such-dir/brz.log'$")

    def test__open_brz_log_ignores_cache_dir_error(self):
        self.overrideAttr(sys, 'stderr', StringIO())
        self.overrideEnv('BRZ_LOG', None)
        self.overrideEnv('BRZ_HOME', '/no-such-dir')
        self.overrideEnv('XDG_CACHE_HOME', '/no-such-dir')
        self.overrideAttr(trace, '_brz_log_filename')
        logf = trace._open_brz_log()
        if os.path.isdir('/no-such-dir'):
            raise TestSkipped('directory creation succeeded')
        self.assertIs(None, logf)
        self.assertContainsRe(sys.stderr.getvalue(), "failed to open trace file: .* '/no-such-dir'$")