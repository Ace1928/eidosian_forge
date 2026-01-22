from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.core import resources
def WaitForCreateKrmApiHostOperation(operation, progress_message='Waiting for cluster to create', max_wait_ms=_MAX_WAIT_TIME_MS):
    """Waits for the given "create" LRO to complete.

  Args:
    operation: the operation to poll.
    progress_message: the message to display while waiting for the operation.
    max_wait_ms: number of ms to wait before raising TimeoutError.

  Raises:
    apitools.base.py.HttpError: if the request returns an HTTP error.

  Returns:
    A messages.KrmApiHost resource.
  """
    client = GetClientInstance()
    operation_ref = resources.REGISTRY.ParseRelativeName(operation.name, collection='krmapihosting.projects.locations.operations')
    poller = waiter.CloudOperationPollerNoResources(client.projects_locations_operations)
    result = waiter.WaitFor(poller, operation_ref, progress_message, max_wait_ms=max_wait_ms, wait_ceiling_ms=_WAIT_CEILING_MS)
    json = encoding.MessageToJson(result)
    messages = GetMessagesModule()
    return encoding.JsonToMessage(messages.KrmApiHost, json)