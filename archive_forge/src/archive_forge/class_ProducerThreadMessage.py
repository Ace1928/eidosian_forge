from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import os
import threading
from apitools.base.py.exceptions import Error as apitools_service_error
from six.moves.http_client import error as six_service_error
class ProducerThreadMessage(StatusMessage):
    """Message class for results obtained by calculations made on ProducerThread.

  It estimates the number of objects and total size currently dealty by
  task_queue. If the task_queue cannot support all objects at once, the
  SeekAheadThread will be responsible for sending an accurate message.
  """

    def __init__(self, num_objects, size, message_time, finished=False):
        """Creates a SeekAheadMessage.

    Args:
      num_objects: Number of total objects that the task_queue has.
      size: Total size corresponding to the sum of the size of objects iterated
          by the task_queue
      message_time: Float representing when message was created (seconds since
          Epoch).
      finished: Boolean to indicate whether this is the final message from the
          ProducerThread. The difference is that this message displays
          the correct total size and number of objects, whereas the
          previous ones were periodic (on the number of files) updates.
    """
        super(ProducerThreadMessage, self).__init__(message_time)
        self.num_objects = num_objects
        self.size = size
        self.finished = finished

    def __str__(self):
        """Returns a string with a valid constructor for this message."""
        return '%s(%s, %s, %s, finished=%s)' % (self.__class__.__name__, self.num_objects, self.size, self.time, self.finished)