import errno
import inspect
import sys
from .. import builtins, commands, config, errors, option, tests, trace
from ..commands import display_command
from . import TestSkipped
class TestRegisterLazy(tests.TestCase):

    def setUp(self):
        super().setUp()
        import breezy.tests.fake_command
        del sys.modules['breezy.tests.fake_command']
        global lazy_command_imported
        lazy_command_imported = False
        commands.install_bzr_command_hooks()

    @staticmethod
    def remove_fake():
        commands.plugin_cmds.remove('fake')

    def assertIsFakeCommand(self, cmd_obj):
        from breezy.tests.fake_command import cmd_fake
        self.assertIsInstance(cmd_obj, cmd_fake)

    def test_register_lazy(self):
        """Ensure lazy registration works"""
        commands.plugin_cmds.register_lazy('cmd_fake', [], 'breezy.tests.fake_command')
        self.addCleanup(self.remove_fake)
        self.assertFalse(lazy_command_imported)
        fake_instance = commands.get_cmd_object('fake')
        self.assertTrue(lazy_command_imported)
        self.assertIsFakeCommand(fake_instance)

    def test_get_unrelated_does_not_import(self):
        commands.plugin_cmds.register_lazy('cmd_fake', [], 'breezy.tests.fake_command')
        self.addCleanup(self.remove_fake)
        commands.get_cmd_object('status')
        self.assertFalse(lazy_command_imported)

    def test_aliases(self):
        commands.plugin_cmds.register_lazy('cmd_fake', ['fake_alias'], 'breezy.tests.fake_command')
        self.addCleanup(self.remove_fake)
        fake_instance = commands.get_cmd_object('fake_alias')
        self.assertIsFakeCommand(fake_instance)