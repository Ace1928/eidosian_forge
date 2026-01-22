from mx import DateTime
import os
import pdb
import sys
import traceback
from google.apputils import app
import gflags as flags
def ReplacementAppUsage(shorthelp=0, writeto_stdout=1, detailed_error=None, exitcode=None):
    AppcommandsUsage(shorthelp, writeto_stdout, detailed_error, exitcode=1, show_cmd=self._command_name, show_global_flags=True)