import errno
import inspect
import sys
from .. import builtins, commands, config, errors, option, tests, trace
from ..commands import display_command
from . import TestSkipped
def test_sorted(self):
    """_see_also is sorted by get_see_also."""
    command = self._get_command_with_see_also(['foo', 'bar'])
    self.assertEqual(['bar', 'foo'], command.get_see_also())