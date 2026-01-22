from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.calliope import base
from googlecloudsdk.core import resources
from googlecloudsdk.core.console import progress_tracker
def WaitForCreatePreviewOperation(operation, progress_message='Creating the preview'):
    """Waits for the given "create preview" LRO to complete.

  Args:
    operation: the operation to poll.
    progress_message: string to display for default progress_tracker.

  Raises:
    apitools.base.py.HttpError: if the request returns an HTTP error.

  Returns:
    A messages.Preview resource.
  """
    client = GetClientInstance()
    operation_ref = resources.REGISTRY.ParseRelativeName(operation.name, collection='config.projects.locations.operations')
    poller = waiter.CloudOperationPoller(client.projects_locations_previews, client.projects_locations_operations)
    poller.detailed_message = ''

    def TrackerUpdateFunc(tracker, result, unused_status):
        """Updates the progress tracker with the result of the operation.

    Args:
      tracker: The ProgressTracker for the operation.
      result: the operation poll result.
      unused_status: map of stages with key as stage key (string) and value is
        the progress_tracker.Stage.
    """
        messages = GetMessagesModule()
        json_val = encoding.MessageToJson(result.metadata)
        preview_metadata = encoding.JsonToMessage(messages.OperationMetadata, json_val).previewMetadata
        logs = ''
        step = ''
        if preview_metadata is not None:
            logs = preview_metadata.logs
            step = preview_metadata.step
        if logs is not None and step is None:
            poller.detailed_message = 'logs={0} '.format(logs)
        elif logs is not None and step is not None:
            poller.detailed_message = 'logs={0}, step={1} '.format(logs, step)
        tracker.Tick()

    def DetailMessageCallback():
        """Returns the detailed progress message to be updated on the progress tracker."""
        return poller.detailed_message
    aborted_message = 'Aborting wait for operation {0}.\n'.format(operation_ref)
    custom_tracker = progress_tracker.ProgressTracker(message=progress_message, detail_message_callback=DetailMessageCallback, aborted_message=aborted_message)
    result = waiter.WaitFor(poller, operation_ref, progress_message, custom_tracker=custom_tracker, tracker_update_func=TrackerUpdateFunc, max_wait_ms=_MAX_WAIT_TIME_MS, wait_ceiling_ms=_WAIT_CEILING_MS)
    return result