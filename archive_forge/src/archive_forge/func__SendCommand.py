from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import locale
import os
import re
import signal
import subprocess
from googlecloudsdk.core.util import encoding
import six
def _SendCommand(self, command):
    """Sends command to the coshell for execution."""
    try:
        self._shell.stdin.write(self._Encode(command + '\n'))
        self._shell.stdin.flush()
    except (IOError, OSError, ValueError):
        self._Exited()