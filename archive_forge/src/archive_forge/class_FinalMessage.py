from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import os
import threading
from apitools.base.py.exceptions import Error as apitools_service_error
from six.moves.http_client import error as six_service_error
class FinalMessage(StatusMessage):
    """Creates a FinalMessage.

  A FinalMessage simply indicates that we have finished our operation.
  """

    def __init__(self, message_time):
        """Creates a FinalMessage.

    Args:
      message_time: Float representing when message was created (seconds since
          Epoch).
    """
        super(FinalMessage, self).__init__(message_time)