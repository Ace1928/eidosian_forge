import errno
import inspect
import sys
from .. import builtins, commands, config, errors, option, tests, trace
from ..commands import display_command
from . import TestSkipped
class TestExtendCommandHook(tests.TestCase):

    def test_fires_on_get_cmd_object(self):
        hook_calls = []
        commands.install_bzr_command_hooks()
        commands.Command.hooks.install_named_hook('extend_command', hook_calls.append, None)

        class cmd_test_extend_command_hook(commands.Command):
            __doc__ = 'A sample command.'
        self.assertEqual([], hook_calls)
        try:
            commands.builtin_command_registry.register(cmd_test_extend_command_hook)
            self.assertEqual([], hook_calls)
            cmd = commands.get_cmd_object('test-extend-command-hook')
            self.assertSubset([cmd], hook_calls)
            del hook_calls[:]
        finally:
            commands.builtin_command_registry.remove('test-extend-command-hook')
        try:
            commands.plugin_cmds.register_lazy('cmd_fake', [], 'breezy.tests.fake_command')
            self.assertEqual([], hook_calls)
            cmd = commands.get_cmd_object('fake')
            self.assertEqual([cmd], hook_calls)
        finally:
            commands.plugin_cmds.remove('fake')