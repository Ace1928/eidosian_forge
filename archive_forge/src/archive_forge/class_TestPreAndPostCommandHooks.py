import errno
import inspect
import sys
from .. import builtins, commands, config, errors, option, tests, trace
from ..commands import display_command
from . import TestSkipped
class TestPreAndPostCommandHooks(tests.TestCase):

    class TestError(Exception):
        __doc__ = 'A test exception.'

    def test_pre_and_post_hooks(self):
        hook_calls = []

        def pre_command(cmd):
            self.assertEqual([], hook_calls)
            hook_calls.append('pre')

        def post_command(cmd):
            self.assertEqual(['pre', 'run'], hook_calls)
            hook_calls.append('post')

        def run(cmd):
            self.assertEqual(['pre'], hook_calls)
            hook_calls.append('run')
        self.overrideAttr(builtins.cmd_rocks, 'run', run)
        commands.install_bzr_command_hooks()
        commands.Command.hooks.install_named_hook('pre_command', pre_command, None)
        commands.Command.hooks.install_named_hook('post_command', post_command, None)
        self.assertEqual([], hook_calls)
        self.run_bzr(['rocks', '-Oxx=12', '-Oyy=foo'])
        self.assertEqual(['pre', 'run', 'post'], hook_calls)

    def test_post_hook_provided_exception(self):
        hook_calls = []

        def post_command(cmd):
            hook_calls.append('post')

        def run(cmd):
            hook_calls.append('run')
            raise self.TestError()
        self.overrideAttr(builtins.cmd_rocks, 'run', run)
        commands.install_bzr_command_hooks()
        commands.Command.hooks.install_named_hook('post_command', post_command, None)
        self.assertEqual([], hook_calls)
        self.assertRaises(self.TestError, commands.run_bzr, ['rocks'])
        self.assertEqual(['run', 'post'], hook_calls)

    def test_pre_command_error(self):
        """Ensure an CommandError in pre_command aborts the command"""
        hook_calls = []

        def pre_command(cmd):
            hook_calls.append('pre')
            raise errors.CommandError()

        def post_command(cmd, e):
            self.fail('post_command should not be called')

        def run(cmd):
            self.fail('command should not be called')
        self.overrideAttr(builtins.cmd_rocks, 'run', run)
        commands.install_bzr_command_hooks()
        commands.Command.hooks.install_named_hook('pre_command', pre_command, None)
        commands.Command.hooks.install_named_hook('post_command', post_command, None)
        self.assertEqual([], hook_calls)
        self.assertRaises(errors.CommandError, commands.run_bzr, ['rocks'])
        self.assertEqual(['pre'], hook_calls)