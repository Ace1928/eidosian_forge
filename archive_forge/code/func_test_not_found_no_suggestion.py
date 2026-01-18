import errno
import inspect
import sys
from .. import builtins, commands, config, errors, option, tests, trace
from ..commands import display_command
from . import TestSkipped
def test_not_found_no_suggestion(self):
    e = self.assertRaises(errors.CommandError, commands.get_cmd_object, 'idontexistand')
    self.assertEqual('unknown command "idontexistand"', str(e))