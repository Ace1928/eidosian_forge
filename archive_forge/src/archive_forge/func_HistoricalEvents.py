from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.cloudnumberregistry.v1alpha import cloudnumberregistry_v1alpha_messages as messages
def HistoricalEvents(self, request, global_params=None):
    """Shows HistoricalEvents in a given registry book.

      Args:
        request: (CloudnumberregistryProjectsLocationsRegistryBooksHistoricalEventsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ShowHistoricalEventsResponse) The response message.
      """
    config = self.GetMethodConfig('HistoricalEvents')
    return self._RunMethod(config, request, global_params=global_params)