from mx import DateTime
import os
import pdb
import sys
import traceback
from google.apputils import app
import gflags as flags
def GetCommandList():
    """Return list of registered commands."""
    global _cmd_list
    return _cmd_list