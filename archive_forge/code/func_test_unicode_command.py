import errno
import inspect
import sys
from .. import builtins, commands, config, errors, option, tests, trace
from ..commands import display_command
from . import TestSkipped
def test_unicode_command(self):
    self.assertRaises(errors.CommandError, commands.run_bzr, ['cmdµ'])