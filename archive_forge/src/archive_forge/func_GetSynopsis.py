from mx import DateTime
import os
import pdb
import sys
import traceback
from google.apputils import app
import gflags as flags
def GetSynopsis():
    """Get synopsis for program.

  Returns:
    Synopsis including program basename.
  """
    return '%s [--global_flags] <command> [--command_flags] [args]' % GetAppBasename()