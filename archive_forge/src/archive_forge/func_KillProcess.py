from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import logging
import os
import re
import signal
import sys
import traceback
from gslib import metrics
from gslib.exception import ControlCException
from gslib.utils.constants import UTF8
from gslib.utils.system_util import IS_WINDOWS
def KillProcess(pid):
    """Make best effort to kill the given process.

  We ignore all exceptions so a caller looping through a list of processes will
  continue attempting to kill each, even if one encounters a problem.

  Args:
    pid: The process ID.
  """
    try:
        if IS_WINDOWS and (3, 0) <= sys.version_info[:3] < (3, 2):
            kernel32 = ctypes.windll.kernel32
            handle = kernel32.OpenProcess(1, 0, pid)
            kernel32.TerminateProcess(handle, 0)
        else:
            os.kill(pid, signal.SIGKILL)
    except:
        pass