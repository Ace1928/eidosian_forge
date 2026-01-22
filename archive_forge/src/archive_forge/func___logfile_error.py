import sys
from winappdbg import win32
from winappdbg import compat
from winappdbg.util import StaticClass
import re
import time
import struct
import traceback
def __logfile_error(self, e):
    """
        Shows an error message to standard error
        if the log file can't be written to.

        Used internally.

        @type  e: Exception
        @param e: Exception raised when trying to write to the log file.
        """
    from sys import stderr
    msg = 'Warning, error writing log file %s: %s\n'
    msg = msg % (self.logfile, str(e))
    stderr.write(DebugLog.log_text(msg))
    self.logfile = None
    self.fd = None