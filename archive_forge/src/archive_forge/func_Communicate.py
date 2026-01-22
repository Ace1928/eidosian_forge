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
def Communicate(self, args, quote=True):
    """Runs args and returns the list of output lines, up to first empty one.

    Args:
      args: The list of command line arguments.
      quote: Shell quote args if True.

    Returns:
      The list of output lines from command args up to the first empty line.
    """
    if quote:
        command = ' '.join([self._Quote(arg) for arg in args])
    else:
        command = ' '.join(args)
    self._SendCommand(command + '\n')
    lines = []
    while True:
        try:
            line = self._ReadLine()
        except (IOError, OSError, ValueError):
            self._Exited()
        if not line:
            break
        lines.append(line)
    return lines