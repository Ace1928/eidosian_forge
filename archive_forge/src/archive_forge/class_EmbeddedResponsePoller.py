from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
class EmbeddedResponsePoller(waiter.CloudOperationPoller):
    """As CloudOperationPoller for polling, but uses the Operation.response."""

    def __init__(self, operation_service):
        self.operation_service = operation_service

    def GetResult(self, operation):
        return operation.response