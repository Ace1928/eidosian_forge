import re
from .. import bzr, commands, controldir, errors, option, registry
from ..builtins import cmd_commit
from ..bzr import knitrepo
from ..commands import parse_args
from . import TestCase
class OptionTests(TestCase):
    """Command-line option tests"""

    def test_parse_args(self):
        """Option parser"""
        self.assertEqual(([], {'author': [], 'exclude': [], 'fixes': [], 'help': True, 'bugs': []}), parse_args(cmd_commit(), ['--help']))
        self.assertEqual(([], {'author': [], 'exclude': [], 'fixes': [], 'message': 'biter', 'bugs': []}), parse_args(cmd_commit(), ['--message=biter']))

    def test_no_more_opts(self):
        """Terminated options"""
        self.assertEqual((['-file-with-dashes'], {'author': [], 'exclude': [], 'fixes': [], 'bugs': []}), parse_args(cmd_commit(), ['--', '-file-with-dashes']))

    def test_option_help(self):
        """Options have help strings."""
        out, err = self.run_bzr('commit --help')
        self.assertContainsRe(out, '--file(.|\\n)*Take commit message from this file\\.')
        self.assertContainsRe(out, '-h.*--help')

    def test_option_help_global(self):
        """Global options have help strings."""
        out, err = self.run_bzr('help status')
        self.assertContainsRe(out, '--show-ids.*Show internal object.')

    def test_option_help_global_hidden(self):
        """Hidden global options have no help strings."""
        out, err = self.run_bzr('help log')
        self.assertNotContainsRe(out, '--message')

    def test_option_arg_help(self):
        """Help message shows option arguments."""
        out, err = self.run_bzr('help commit')
        self.assertEqual(err, '')
        self.assertContainsRe(out, '--file[ =]MSGFILE')

    def test_unknown_short_opt(self):
        out, err = self.run_bzr('help -r', retcode=3)
        self.assertContainsRe(err, 'no such option')

    def test_set_short_name(self):
        o = option.Option('wiggle')
        o.set_short_name('w')
        self.assertEqual(o.short_name(), 'w')

    def test_allow_dash(self):
        """Test that we can pass a plain '-' as an argument."""
        self.assertEqual(['-'], parse_args(cmd_commit(), ['-'])[0])

    def parse(self, options, args):
        parser = option.get_optparser(options)
        return parser.parse_args(args)

    def test_conversion(self):
        options = [option.Option('hello')]
        opts, args = self.parse(options, ['--no-hello', '--hello'])
        self.assertEqual(True, opts.hello)
        opts, args = self.parse(options, [])
        self.assertFalse(hasattr(opts, 'hello'))
        opts, args = self.parse(options, ['--hello', '--no-hello'])
        self.assertEqual(False, opts.hello)
        options = [option.Option('number', type=int)]
        opts, args = self.parse(options, ['--number', '6'])
        self.assertEqual(6, opts.number)
        self.assertRaises(errors.CommandError, self.parse, options, ['--number'])
        self.assertRaises(errors.CommandError, self.parse, options, ['--no-number'])
        self.assertRaises(errors.CommandError, self.parse, options, ['--number', 'a'])

    def test_is_hidden(self):
        self.assertTrue(option.Option('foo', hidden=True).is_hidden('foo'))
        self.assertFalse(option.Option('foo', hidden=False).is_hidden('foo'))

    def test_registry_conversion(self):
        registry = controldir.ControlDirFormatRegistry()
        bzr.register_metadir(registry, 'one', 'RepositoryFormat7', 'one help')
        bzr.register_metadir(registry, 'two', 'RepositoryFormatKnit1', 'two help')
        bzr.register_metadir(registry, 'hidden', 'RepositoryFormatKnit1', 'two help', hidden=True)
        registry.set_default('one')
        options = [option.RegistryOption('format', '', registry, str)]
        opts, args = self.parse(options, ['--format', 'one'])
        self.assertEqual({'format': 'one'}, opts)
        opts, args = self.parse(options, ['--format', 'two'])
        self.assertEqual({'format': 'two'}, opts)
        self.assertRaises(option.BadOptionValue, self.parse, options, ['--format', 'three'])
        self.assertRaises(errors.CommandError, self.parse, options, ['--two'])
        options = [option.RegistryOption('format', '', registry, str, value_switches=True)]
        opts, args = self.parse(options, ['--two'])
        self.assertEqual({'format': 'two'}, opts)
        opts, args = self.parse(options, ['--two', '--one'])
        self.assertEqual({'format': 'one'}, opts)
        opts, args = self.parse(options, ['--two', '--one', '--format', 'two'])
        self.assertEqual({'format': 'two'}, opts)
        options = [option.RegistryOption('format', '', registry, str, enum_switch=False)]
        self.assertRaises(errors.CommandError, self.parse, options, ['--format', 'two'])

    def test_override(self):
        options = [option.Option('hello', type=str), option.Option('hi', type=str, param_name='hello')]
        opts, args = self.parse(options, ['--hello', 'a', '--hello', 'b'])
        self.assertEqual('b', opts.hello)
        opts, args = self.parse(options, ['--hello', 'b', '--hello', 'a'])
        self.assertEqual('a', opts.hello)
        opts, args = self.parse(options, ['--hello', 'a', '--hi', 'b'])
        self.assertEqual('b', opts.hello)
        opts, args = self.parse(options, ['--hi', 'b', '--hello', 'a'])
        self.assertEqual('a', opts.hello)

    def test_registry_converter(self):
        options = [option.RegistryOption('format', '', controldir.format_registry, controldir.format_registry.make_controldir)]
        opts, args = self.parse(options, ['--format', 'knit'])
        self.assertIsInstance(opts.format.repository_format, knitrepo.RepositoryFormatKnit1)

    def test_lazy_registry(self):
        options = [option.RegistryOption('format', '', lazy_registry=('breezy.controldir', 'format_registry'), converter=str)]
        opts, args = self.parse(options, ['--format', 'knit'])
        self.assertEqual({'format': 'knit'}, opts)
        self.assertRaises(option.BadOptionValue, self.parse, options, ['--format', 'BAD'])

    def test_from_kwargs(self):
        my_option = option.RegistryOption.from_kwargs('my-option', help='test option', short='be short', be_long='go long')
        self.assertEqual(['my-option'], [x[0] for x in my_option.iter_switches()])
        my_option = option.RegistryOption.from_kwargs('my-option', help='test option', title='My option', short='be short', be_long='go long', value_switches=True)
        self.assertEqual(['my-option', 'be-long', 'short'], [x[0] for x in my_option.iter_switches()])
        self.assertEqual('test option', my_option.help)

    def test_help(self):
        registry = controldir.ControlDirFormatRegistry()
        bzr.register_metadir(registry, 'one', 'RepositoryFormat7', 'one help')
        bzr.register_metadir(registry, 'two', 'breezy.bzr.knitrepo.RepositoryFormatKnit1', 'two help')
        bzr.register_metadir(registry, 'hidden', 'RepositoryFormat7', 'hidden help', hidden=True)
        registry.set_default('one')
        options = [option.RegistryOption('format', 'format help', registry, str, value_switches=True, title='Formats')]
        parser = option.get_optparser(options)
        value = parser.format_option_help()
        self.assertContainsRe(value, 'format.*format help')
        self.assertContainsRe(value, 'one.*one help')
        self.assertContainsRe(value, 'Formats:\n *--format')
        self.assertNotContainsRe(value, 'hidden help')

    def test_iter_switches(self):
        opt = option.Option('hello', help='fg')
        self.assertEqual(list(opt.iter_switches()), [('hello', None, None, 'fg')])
        opt = option.Option('hello', help='fg', type=int)
        self.assertEqual(list(opt.iter_switches()), [('hello', None, 'ARG', 'fg')])
        opt = option.Option('hello', help='fg', type=int, argname='gar')
        self.assertEqual(list(opt.iter_switches()), [('hello', None, 'GAR', 'fg')])
        registry = controldir.ControlDirFormatRegistry()
        bzr.register_metadir(registry, 'one', 'RepositoryFormat7', 'one help')
        bzr.register_metadir(registry, 'two', 'breezy.bzr.knitrepo.RepositoryFormatKnit1', 'two help')
        registry.set_default('one')
        opt = option.RegistryOption('format', 'format help', registry, value_switches=False)
        self.assertEqual(list(opt.iter_switches()), [('format', None, 'ARG', 'format help')])
        opt = option.RegistryOption('format', 'format help', registry, value_switches=True)
        self.assertEqual(list(opt.iter_switches()), [('format', None, 'ARG', 'format help'), ('default', None, None, 'one help'), ('one', None, None, 'one help'), ('two', None, None, 'two help')])

    def test_option_callback_bool(self):
        """Test booleans get True and False passed correctly to a callback."""
        cb_calls = []

        def cb(option, name, value, parser):
            cb_calls.append((option, name, value, parser))
        options = [option.Option('hello', custom_callback=cb)]
        opts, args = self.parse(options, ['--hello', '--no-hello'])
        self.assertEqual(2, len(cb_calls))
        opt, name, value, parser = cb_calls[0]
        self.assertEqual('hello', name)
        self.assertTrue(value)
        opt, name, value, parser = cb_calls[1]
        self.assertEqual('hello', name)
        self.assertFalse(value)

    def test_option_callback_str(self):
        """Test callbacks work for string options both long and short."""
        cb_calls = []

        def cb(option, name, value, parser):
            cb_calls.append((option, name, value, parser))
        options = [option.Option('hello', type=str, custom_callback=cb, short_name='h')]
        opts, args = self.parse(options, ['--hello', 'world', '-h', 'mars'])
        self.assertEqual(2, len(cb_calls))
        opt, name, value, parser = cb_calls[0]
        self.assertEqual('hello', name)
        self.assertEqual('world', value)
        opt, name, value, parser = cb_calls[1]
        self.assertEqual('hello', name)
        self.assertEqual('mars', value)