from mx import DateTime
import os
import pdb
import sys
import traceback
from google.apputils import app
import gflags as flags
def GetMaxCommandLength():
    """Returns the length of the longest registered command."""
    return max([len(cmd_name) for cmd_name in GetCommandList()])