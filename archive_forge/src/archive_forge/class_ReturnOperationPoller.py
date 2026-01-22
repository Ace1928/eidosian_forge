from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.resource_manager import tags
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import resources
class ReturnOperationPoller(waiter.CloudOperationPoller):
    """Polls for operations that retrieve the operation rather than the resource.

  This is needed for Delete operations, where the response is Empty. It is also
  needed for services that do not have a Get* method, such as TagBindings.
  """

    def __init__(self, operation_service):
        """Sets up poller for polling operations.

    Args:
      operation_service: apitools.base.py.base_api.BaseApiService, api service
        for retrieving information about ongoing operation.
    """
        self.operation_service = operation_service

    def GetResult(self, operation):
        """Overrides.

    Response for Deletion Operation is of type google.protobuf.Empty and hence
    we can return the operation itself as the result. For operations without a
    Get[Resource] method, we have no choice but to return the operation.

    Args:
      operation: api_name_messages.Operation.

    Returns:
      operation
    """
        return operation