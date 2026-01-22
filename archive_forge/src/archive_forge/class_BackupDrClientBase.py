from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import datetime
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.core import resources
class BackupDrClientBase(object):
    """Base class for Backup and DR API client wrappers."""

    def __init__(self, api_version=DEFAULT_API_VERSION):
        self._client = apis.GetClientInstance('backupdr', api_version)
        self._messages = apis.GetMessagesModule('backupdr', api_version)
        self.service = None
        self.operations_service = self.client.projects_locations_operations

    @property
    def client(self):
        return self._client

    @property
    def messages(self):
        return self._messages

    def GetOperationRef(self, operation):
        """Converts an Operation to a Resource that can be used with `waiter.WaitFor`."""
        return resources.REGISTRY.ParseRelativeName(operation.name, collection='backupdr.projects.locations.operations')

    def WaitForOperation(self, operation_ref, message, has_result=True, max_wait=datetime.timedelta(seconds=3600)):
        """Waits for an operation to complete.

    Polls the Backup and DR Operation service until the operation completes,
    fails, or
    max_wait_seconds elapses.

    Args:
      operation_ref: a Resource created by GetOperationRef describing the
        operation.
      message: the message to display to the user while they wait.
      has_result: if True, the function will return the target of the operation
        when it completes. If False, nothing will be returned (useful for Delete
        operations)
      max_wait: The time to wait for the operation to succeed before returning.

    Returns:
      if has_result = True, a Backup and DR entity.
      Otherwise, None.
    """
        if has_result:
            poller = waiter.CloudOperationPoller(self.service, self.operations_service)
        else:
            poller = waiter.CloudOperationPollerNoResources(self.operations_service)
        return waiter.WaitFor(poller, operation_ref, message, max_wait_ms=max_wait.seconds * 1000)