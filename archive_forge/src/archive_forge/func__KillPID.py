from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import contextlib
import errno
import os
import re
import signal
import subprocess
import sys
import threading
import time
from googlecloudsdk.core import argv_utils
from googlecloudsdk.core import config
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.configurations import named_configs
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import parallel
from googlecloudsdk.core.util import platforms
import six
from six.moves import map
def _KillPID(pid):
    """Kills the given process with SIGTERM, then with SIGKILL if it doesn't stop.

  Args:
    pid: The process id of the process to check.
  """
    try:
        os.kill(pid, signal.SIGTERM)
        deadline = time.time() + 3
        while time.time() < deadline:
            if not _IsStillRunning(pid):
                return
            time.sleep(0.1)
        os.kill(pid, signal.SIGKILL)
    except OSError as error:
        if 'No such process' not in error.strerror:
            exceptions.reraise(sys.exc_info()[1])