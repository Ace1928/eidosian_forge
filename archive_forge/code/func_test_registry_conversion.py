import re
from .. import bzr, commands, controldir, errors, option, registry
from ..builtins import cmd_commit
from ..bzr import knitrepo
from ..commands import parse_args
from . import TestCase
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