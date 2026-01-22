from mx import DateTime
import os
import pdb
import sys
import traceback
from google.apputils import app
import gflags as flags
def _UsageFooter(detailed_error, cmd_names):
    """Output a footer at the end of usage or help output.

  Args:
    detailed_error: additional detail about why usage info was presented.
    cmd_names:      list of command names for which help was shown or None.
  Returns:
    Generated footer that contains 'Run..' messages if appropriate.
  """
    footer = []
    if not cmd_names or len(cmd_names) == 1:
        footer.append("Run '%s help' to see the list of available commands." % GetAppBasename())
    if not cmd_names or len(cmd_names) == len(GetCommandList()):
        footer.append("Run '%s help <command>' to get help for <command>." % GetAppBasename())
    if detailed_error is not None:
        if footer:
            footer.append('')
        footer.append('%s' % detailed_error)
    return '\n'.join(footer)