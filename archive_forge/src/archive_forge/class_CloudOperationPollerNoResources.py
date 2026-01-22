from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import time
from apitools.base.py import encoding
from googlecloudsdk.core import exceptions
from googlecloudsdk.core.console import progress_tracker
from googlecloudsdk.core.util import retry
import six
class CloudOperationPollerNoResources(OperationPoller):
    """Manages longrunning Operations for Cloud API that creates no resources.

  See https://cloud.google.com/speech/reference/rpc/google.longrunning
  """

    def __init__(self, operation_service, get_name_func=None):
        """Sets up poller for cloud operations.

    Args:
      operation_service: apitools.base.py.base_api.BaseApiService, api service
        for retrieving information about ongoing operation.

        Note that the operation_service Get request must have a
        single attribute called 'name'.
      get_name_func: the function to use to get the name from the operation_ref.
        This is to allow polling with non-traditional operation resource names.
        If the resource name is compatible with gcloud parsing, use
        `lambda x: x.RelativeName()`.
    """
        self.operation_service = operation_service
        self.get_name = get_name_func or (lambda x: x.RelativeName())

    def IsDone(self, operation):
        """Overrides."""
        if operation.done:
            if operation.error:
                raise OperationError(operation.error.message)
            return True
        return False

    def Poll(self, operation_ref):
        """Overrides.

    Args:
      operation_ref: googlecloudsdk.core.resources.Resource.

    Returns:
      fetched operation message.
    """
        request_type = self.operation_service.GetRequestType('Get')
        return self.operation_service.Get(request_type(name=self.get_name(operation_ref)))

    def GetResult(self, operation):
        """Overrides to get the response from the completed operation.

    Args:
      operation: api_name_messages.Operation.

    Returns:
      the 'response' field of the Operation.
    """
        return operation.response