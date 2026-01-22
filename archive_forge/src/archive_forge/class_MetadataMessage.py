from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import os
import threading
from apitools.base.py.exceptions import Error as apitools_service_error
from six.moves.http_client import error as six_service_error
class MetadataMessage(StatusMessage):
    """Creates a MetadataMessage.

  A MetadataMessage simply indicates that a metadata operation on a given object
  has been successfully done. The only passed argument is the time when such
  operation has finished.
  """

    def __init__(self, message_time):
        """Creates a MetadataMessage.

    Args:
      message_time: Float representing when message was created (seconds since
          Epoch).
    """
        super(MetadataMessage, self).__init__(message_time)