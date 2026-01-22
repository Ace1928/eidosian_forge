from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import os
import threading
from apitools.base.py.exceptions import Error as apitools_service_error
from six.moves.http_client import error as six_service_error
class StatusMessage(object):
    """General StatusMessage class.

  All Message classes inherit this StatusMessage class.
  """

    def __init__(self, message_time, process_id=None, thread_id=None):
        """Creates a Message.

    Args:
      message_time: Time that this message was created (since Epoch).
      process_id: Process ID that produced this message (overridable for
          testing).
      thread_id: Thread ID that produced this message (overridable for testing).
    """
        self.time = message_time
        self.process_id = process_id or os.getpid()
        self.thread_id = thread_id or threading.current_thread().ident

    def __str__(self):
        """Returns a string with a valid constructor for this message."""
        return '%s(%s, process_id=%s, thread_id=%s)' % (self.__class__.__name__, self.time, self.process_id, self.thread_id)