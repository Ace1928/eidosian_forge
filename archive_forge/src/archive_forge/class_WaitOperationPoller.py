from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.command_lib.run import exceptions as serverless_exceptions
from googlecloudsdk.core import exceptions
class WaitOperationPoller(waiter.CloudOperationPoller):
    """Poll for a long running operation using Wait instead of Get."""

    def Poll(self, operation_ref):
        """Overrides.

    Args:
      operation_ref: googlecloudsdk.core.resources.Resource.

    Returns:
      fetched operation message.
    """
        request_type = self.operation_service.GetRequestType('Wait')
        return self.operation_service.Wait(request_type(name=operation_ref.RelativeName()))