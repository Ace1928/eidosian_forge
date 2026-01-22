from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
import signal
import sys
from googlecloudsdk.core import log
def HandleInterrupt(signal_number=None, frame=None):
    """Handles keyboard interrupts (aka SIGINT, ^C).

  Disables the stack trace when a command is killed by keyboard interrupt.

  Args:
    signal_number: The interrupt signal number.
    frame: The signal stack frame context.
  """
    del signal_number, frame
    message = '\n\nCommand killed by keyboard interrupt\n'
    try:
        log.err.Print(message)
    except NameError:
        sys.stderr.write(message)
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    os.kill(os.getpid(), signal.SIGINT)
    sys.exit(1)