import errno
import inspect
import sys
from .. import builtins, commands, config, errors, option, tests, trace
from ..commands import display_command
from . import TestSkipped
class cmd_foo(commands.Command):
    __doc__ = 'Bar'
    takes_options = options