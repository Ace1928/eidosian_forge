from mx import DateTime
import os
import pdb
import sys
import traceback
from google.apputils import app
import gflags as flags
def GetCommand(command_required):
    """Get the command or return None (or issue an error) if there is none.

  Args:
    command_required: whether to issue an error if no command is present

  Returns:
    command or None, if command_required is True then return value is a valid
    command or the program will exit. The program also exits if a command was
    specified but that command does not exist.
  """
    global _cmd_argv
    _cmd_argv = ParseFlagsWithUsage(_cmd_argv)
    if len(_cmd_argv) < 2:
        if command_required:
            ShortHelpAndExit('FATAL Command expected but none given')
        return None
    command = GetCommandByName(_cmd_argv[1])
    if command is None:
        ShortHelpAndExit("FATAL Command '%s' unknown" % _cmd_argv[1])
    del _cmd_argv[1]
    return command