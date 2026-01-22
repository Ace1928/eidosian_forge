from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from googlecloudsdk.api_lib.container.gkemulticloud import client
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.command_lib.container.gkemulticloud import constants
from googlecloudsdk.core import log
from googlecloudsdk.core.console import progress_tracker
class OperationsClient(client.ClientBase):
    """Client for managing LROs."""

    def __init__(self, **kwargs):
        super(OperationsClient, self).__init__(**kwargs)
        self._service = self._client.projects_locations_operations
        self._list_result_field = 'operations'

    def Wait(self, operation_ref, message):
        """Waits for an LRO to complete.

    Args:
      operation_ref: object, passed to operation poller poll method.
      message: str, string to display for the progress tracker.
    """
        poller = _Poller(self._service)
        waiter.WaitFor(poller=poller, operation_ref=operation_ref, custom_tracker=progress_tracker.ProgressTracker(message=message, detail_message_callback=poller.GetDetailMessage, aborted_message='Aborting wait for operation {}.\n'.format(operation_ref)), wait_ceiling_ms=constants.MAX_LRO_POLL_INTERVAL_MS, max_wait_ms=constants.MAX_LRO_WAIT_MS)

    def Cancel(self, operation_ref):
        """Cancels an ongoing LRO.

    Args:
      operation_ref: object, operation resource to be canceled.
    """
        request_type = self._service.GetRequestType('Cancel')
        self._service.Cancel(request_type(name=operation_ref.RelativeName()))