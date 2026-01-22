from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.util import waiter
class IdentityPoolOperationPollerNoResources(waiter.CloudOperationPoller):
    """Manages an identity pool long-running operation that creates no resources."""

    def GetResult(self, operation):
        """Overrides.

    Override the default implementation because Identity Pools
    GetOperation does not return anything in the Operation.response field.

    Args:
      operation: api_name_message.Operation.

    Returns:
      None
    """
        return None