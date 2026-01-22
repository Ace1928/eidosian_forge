from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import contextlib
import os
import random
import re
import socket
import subprocess
import tempfile
from googlecloudsdk.core import config
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import yaml
from googlecloudsdk.core.resource import resource_printer
from googlecloudsdk.core.updater import local_state
from googlecloudsdk.core.updater import update_manager
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import platforms
import portpicker
import six
def _GetLogNo(self):
    """Returns the OS-level handle to log file.

    This handle is the same as would be returned by os.open(). This is what the
    subprocess interface expects. Note that the caller needs to make sure to
    close this to avoid leaking file descriptors.

    Returns:
      int, OS-level handle to log file
    """
    log_file_no, log_file = tempfile.mkstemp()
    log.status.Print('Logging {0} to: {1}'.format(self.service_name, log_file))
    return log_file_no