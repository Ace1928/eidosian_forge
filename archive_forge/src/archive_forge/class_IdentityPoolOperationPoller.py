from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.util import waiter
class IdentityPoolOperationPoller(waiter.CloudOperationPoller):
    """Manages an identity pool long-running operation."""

    def GetResult(self, operation):
        """Overrides.

    Override the default implementation because Identity Pools
    GetOperation does not return anything in the Operation.response field.

    Args:
      operation: api_name_message.Operation.

    Returns:
      result of result_service.Get request.
    """
        request_type = self.result_service.GetRequestType('Get')
        resource_name = '/'.join(operation.name.split('/')[:-2])
        return self.result_service.Get(request_type(name=resource_name))