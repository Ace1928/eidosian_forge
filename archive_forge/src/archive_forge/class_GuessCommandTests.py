import errno
import inspect
import sys
from .. import builtins, commands, config, errors, option, tests, trace
from ..commands import display_command
from . import TestSkipped
class GuessCommandTests(tests.TestCase):

    def setUp(self):
        super().setUp()
        commands._register_builtin_commands()
        commands.install_bzr_command_hooks()

    def test_guess_override(self):
        self.assertEqual('ci', commands.guess_command('ic'))

    def test_guess(self):
        commands.get_cmd_object('status')
        self.assertEqual('status', commands.guess_command('statue'))

    def test_none(self):
        self.assertIs(None, commands.guess_command('nothingisevenclose'))