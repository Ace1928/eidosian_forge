from mx import DateTime
import os
import pdb
import sys
import traceback
from google.apputils import app
import gflags as flags
def Run(self, argv):
    """Execute help command.

    If an argument is given and that argument is a registered command
    name, then help specific to that command is being displayed.
    If the command is unknown then a fatal error will be displayed. If
    no argument is present then help for all commands will be presented.

    If a specific command help is being generated, the list of commands is
    temporarily replaced with one containing only that command. Thus the call
    to usage() will only show help for that command. Otherwise call usage()
    will show help for all registered commands as it sees all commands.

    Args:
      argv: Remaining command line flags and arguments after parsing command
            (that is a copy of sys.argv at the time of the function call with
            all parsed flags removed).
            So argv[0] is the program and argv[1] will be the first argument to
            the call. For instance 'tool.py help command' will result in argv
            containing ('tool.py', 'command'). In this case the list of
            commands is searched for 'command'.

    Returns:
      1 for failure
    """
    if len(argv) > 1 and argv[1] in GetFullCommandList():
        show_cmd = argv[1]
    else:
        show_cmd = None
    AppcommandsUsage(shorthelp=0, writeto_stdout=1, detailed_error=None, exitcode=1, show_cmd=show_cmd, show_global_flags=False)