from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import os
import threading
from apitools.base.py.exceptions import Error as apitools_service_error
from six.moves.http_client import error as six_service_error
class RetryableErrorMessage(StatusMessage):
    """Message class for retryable errors encountered by the JSON API.

  This class contains information about the retryable error encountered to
  report to analytics collection and to display in the UI.
  """

    def __init__(self, exception, message_time, num_retries=0, total_wait_sec=0, process_id=None, thread_id=None):
        """Creates a RetryableErrorMessage.

    Args:
      exception: The retryable error that was thrown.
      message_time: Float representing when message was created (seconds since
          Epoch).
      num_retries: The number of retries consumed so far.
      total_wait_sec: The total amount of time waited so far in retrying.
      process_id: Process ID that produced this message (overridable for
          testing).
      thread_id: Thread ID that produced this message (overridable for testing).
    """
        super(RetryableErrorMessage, self).__init__(message_time, process_id=process_id, thread_id=thread_id)
        self.error_type = exception.__class__.__name__
        if exception.__class__.__module__ in ('socket', '_socket'):
            self.error_type = 'Socket' + exception.__class__.__name__.capitalize()
        if isinstance(exception, apitools_service_error) or isinstance(exception, six_service_error):
            self.is_service_error = True
        else:
            self.is_service_error = False
        self.num_retries = num_retries
        self.total_wait_sec = total_wait_sec

    def __str__(self):
        """Returns a string with a valid constructor for this message."""
        return '%s(%s(), num_retries=%s, total_wait_sec=%s, time=%s, process_id=%s, thread_id=%s)' % (self.__class__.__name__, self.error_type, self.num_retries, self.total_wait_sec, self.time, self.process_id, self.thread_id)