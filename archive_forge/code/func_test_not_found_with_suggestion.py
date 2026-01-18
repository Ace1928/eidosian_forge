import errno
import inspect
import sys
from .. import builtins, commands, config, errors, option, tests, trace
from ..commands import display_command
from . import TestSkipped
def test_not_found_with_suggestion(self):
    e = self.assertRaises(errors.CommandError, commands.get_cmd_object, 'statue')
    self.assertEqual('unknown command "statue". Perhaps you meant "status"', str(e))