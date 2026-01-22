from mx import DateTime
import os
import pdb
import sys
import traceback
from google.apputils import app
import gflags as flags
def AddCmdFunc(command_name, cmd_func, command_aliases=None, all_commands_help=None):
    """Add a new command to the list of registered commands.

  Args:
    command_name:      name of the command which will be used in argument
                       parsing
    cmd_func:          command function, this function received the remaining
                       arguments as its only parameter. It is supposed to do the
                       command work and then return with the command result that
                       is being used as the shell exit code.
    command_aliases:   A list of command aliases that the command can be run as.
    all_commands_help: Help message to be displayed in place of func.__doc__
                       when all commands are displayed.
  """
    _AddCmdInstance(command_name, _FunctionalCmd(command_name, flags.FlagValues(), cmd_func, command_aliases=command_aliases, all_commands_help=all_commands_help), command_aliases)