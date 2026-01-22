from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
from typing import Generator
from apitools.base.py import encoding
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.container.fleet import util
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.calliope import base
from googlecloudsdk.core import resources
from googlecloudsdk.generated_clients.apis.gkehub.v1alpha import gkehub_v1alpha_messages as messages
class OperationClient:
    """Client for the GKE Hub API long-running operations."""

    def __init__(self, release_track: base.ReleaseTrack):
        self.messages = util.GetMessagesModule(release_track)
        self.client = util.GetClientInstance(release_track=release_track)
        self.service = self.client.projects_locations_operations

    def Wait(self, operation_ref: resources.Resource) -> messages.Operation:
        """Waits for a long-running operation to complete.

    Polling message is printed to the terminal periodically, until the operation
    completes or times out.

    Args:
      operation_ref: Long-running peration in the format of resource argument.

    Returns:
      A completed long-running operation.
    """
        return waiter.WaitFor(waiter.CloudOperationPollerNoResources(self.service), operation_ref, 'Waiting for operation [{}] to complete'.format(operation_ref.RelativeName()), wait_ceiling_ms=10000, max_wait_ms=43200000)

    def Describe(self, req: messages.GkehubProjectsLocationsOperationsGetRequest) -> messages.Operation:
        """Describes a long-running operation."""
        return self.client.projects_locations_operations.Get(req)

    def List(self, req: messages.GkehubProjectsLocationsOperationsListRequest, page_size=None, limit=None) -> Generator[messages.Operation, None, None]:
        """Lists long-running operations.

    Currently gcloud implements client-side filtering and limiting behavior.

    Args:
      req: List request to pass to the server.
      page_size: Maximum number of resources per page.
      limit: Client-side limit control.

    Returns:
      A list of long-running operations.
    """
        return list_pager.YieldFromList(self.client.projects_locations_operations, req, field='operations', batch_size=page_size, limit=limit, batch_size_attribute='pageSize')