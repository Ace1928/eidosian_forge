import errno
import inspect
import sys
from .. import builtins, commands, config, errors, option, tests, trace
from ..commands import display_command
from . import TestSkipped
def test_unicode_option(self):
    import optparse
    if optparse.__version__ == '1.5.3':
        raise TestSkipped("optparse 1.5.3 can't handle unicode options")
    self.assertRaises(errors.CommandError, commands.run_bzr, ['log', '--optionµ'])