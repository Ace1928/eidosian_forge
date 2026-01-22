from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.data_fusion import datafusion as df
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.core import exceptions as core_exceptions
class OperationPoller(waiter.CloudOperationPollerNoResources):
    """Class for polling Data Fusion long running Operations."""

    def __init__(self):
        super(OperationPoller, self).__init__(df.Datafusion().client.projects_locations_operations, lambda x: x)

    def IsDone(self, operation):
        if operation.done:
            if operation.error:
                raise OperationError(operation.name, operation.error.message)
            return True
        return False