from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.netapp import util
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.calliope import base
class NetAppClient(object):
    """Wrapper for working with the Cloud NetApp Files API Client."""

    def __init__(self, release_track=base.ReleaseTrack.ALPHA):
        if release_track == base.ReleaseTrack.ALPHA:
            self._adapter = AlphaNetappAdapter()
        elif release_track == base.ReleaseTrack.BETA:
            self._adapter = BetaNetappAdapter()
        elif release_track == base.ReleaseTrack.GA:
            self._adapter = NetappAdapter()
        else:
            raise ValueError('[{}] is not a valid API version.'.format(util.VERSION_MAP[release_track]))

    @property
    def client(self):
        return self._adapter.client

    @property
    def messages(self):
        return self._adapter.messages

    def GetOperation(self, operation_ref):
        """Gets description of a long-running operation.

    Args:
      operation_ref: the operation reference.

    Returns:
      messages.GoogleLongrunningOperation, the operation.
    """
        request = self.messages.NetappProjectsLocationsOperationsGetRequest(name=operation_ref.RelativeName())
        return self.client.projects_locations_operations.Get(request)

    def WaitForOperation(self, operation_ref):
        """Waits on the long-running operation until the done field is True.

    Args:
      operation_ref: the operation reference.

    Raises:
      waiter.OperationError: if the operation contains an error.

    Returns:
      the 'response' field of the Operation.
    """
        return waiter.WaitFor(waiter.CloudOperationPollerNoResources(self.client.projects_locations_operations), operation_ref, 'Waiting for [{0}] to finish'.format(operation_ref.Name()))

    def CancelOperation(self, operation_ref):
        """Cancels a long-running operation.

    Args:
      operation_ref: the operation reference.

    Returns:
      Empty response message.
    """
        request = self.messages.NetappProjectsLocationsOperationsCancelRequest(name=operation_ref.RelativeName())
        return self.client.projects_locations_operations.Cancel(request)

    def GetLocation(self, location_ref):
        request = self.messages.NetappProjectsLocationsGetRequest(name=location_ref)
        return self.client.projects_locations.Get(request)

    def ListLocations(self, project_ref, limit=None):
        request = self.messages.NetappProjectsLocationsListRequest(name=project_ref.RelativeName())
        return list_pager.YieldFromList(self.client.projects_locations, request, field='locations', limit=limit, batch_size_attribute='pageSize')

    def ListOperations(self, location_ref, limit=None):
        """Make API calls to List active Cloud NetApp operations.

    Args:
      location_ref: The parsed location of the listed NetApp resources.
      limit: The number of Cloud NetApp resources to limit the results to. This
        limit is passed to the server and the server does the limiting.

    Returns:
      Generator that yields the Cloud NetApp resources.
    """
        request = self.messages.NetappProjectsLocationsOperationsListRequest(name=location_ref)
        return list_pager.YieldFromList(self.client.projects_locations_operations, request, field='operations', limit=limit, batch_size_attribute='pageSize')