import errno
import inspect
import sys
from .. import builtins, commands, config, errors, option, tests, trace
from ..commands import display_command
from . import TestSkipped
class cmd_test_extend_command_hook(commands.Command):
    __doc__ = 'A sample command.'