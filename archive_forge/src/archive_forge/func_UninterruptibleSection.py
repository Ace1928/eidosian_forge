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
def UninterruptibleSection(stream, message=None):
    """Run a section of code with CTRL-C disabled.

  When in this context manager, the ctrl-c signal is caught and a message is
  printed saying that the action cannot be cancelled.

  Args:
    stream: the stream to write to if SIGINT is received
    message: str, optional: the message to write

  Returns:
    Context manager that is uninterruptible during its lifetime.
  """
    message = '\n\n{message}\n\n'.format(message=message or 'This operation cannot be cancelled.')

    def _Handler(unused_signal, unused_frame):
        stream.write(message)
    return CtrlCSection(_Handler)