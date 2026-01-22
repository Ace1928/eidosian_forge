from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.compute.beta import compute_beta_messages as messages
class InterconnectLocationsService(base_api.BaseApiService):
    """Service class for the interconnectLocations resource."""
    _NAME = 'interconnectLocations'

    def __init__(self, client):
        super(ComputeBeta.InterconnectLocationsService, self).__init__(client)
        self._upload_configs = {}

    def Get(self, request, global_params=None):
        """Returns the details for the specified interconnect location. Gets a list of available interconnect locations by making a list() request.

      Args:
        request: (ComputeInterconnectLocationsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (InterconnectLocation) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.interconnectLocations.get', ordered_params=['project', 'interconnectLocation'], path_params=['interconnectLocation', 'project'], query_params=[], relative_path='projects/{project}/global/interconnectLocations/{interconnectLocation}', request_field='', request_type_name='ComputeInterconnectLocationsGetRequest', response_type_name='InterconnectLocation', supports_download=False)

    def List(self, request, global_params=None):
        """Retrieves the list of interconnect locations available to the specified project.

      Args:
        request: (ComputeInterconnectLocationsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (InterconnectLocationList) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.interconnectLocations.list', ordered_params=['project'], path_params=['project'], query_params=['filter', 'maxResults', 'orderBy', 'pageToken', 'returnPartialSuccess'], relative_path='projects/{project}/global/interconnectLocations', request_field='', request_type_name='ComputeInterconnectLocationsListRequest', response_type_name='InterconnectLocationList', supports_download=False)