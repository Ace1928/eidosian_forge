from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import pprint
import uuid
from apitools.base.py import encoding
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.calliope import base
from googlecloudsdk.core import log
from googlecloudsdk.core import resources
import six
class CloudDmsOperationPoller(waiter.CloudOperationPoller):
    """Manages a longrunning Operations for cloud DMS.

  It is needed since we want to return the entire error rather than just the
  error message as the base class does.

  See https://cloud.google.com/speech/reference/rpc/google.longrunning
  """

    def IsDone(self, operation):
        """Overrides."""
        if operation.done:
            if operation.error:
                op_error = encoding.MessageToDict(operation.error)
                raise waiter.OperationError('\n' + pprint.pformat(op_error))
            return True
        return False