import errno
import inspect
import sys
from .. import builtins, commands, config, errors, option, tests, trace
from ..commands import display_command
from . import TestSkipped
@staticmethod
def _get_command_with_see_also(see_also):

    class ACommand(commands.Command):
        __doc__ = 'A sample command.'
        _see_also = see_also
    return ACommand()