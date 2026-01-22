from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.compute.beta import compute_beta_messages as messages
class InterconnectRemoteLocationsService(base_api.BaseApiService):
    """Service class for the interconnectRemoteLocations resource."""
    _NAME = 'interconnectRemoteLocations'

    def __init__(self, client):
        super(ComputeBeta.InterconnectRemoteLocationsService, self).__init__(client)
        self._upload_configs = {}

    def Get(self, request, global_params=None):
        """Returns the details for the specified interconnect remote location. Gets a list of available interconnect remote locations by making a list() request.

      Args:
        request: (ComputeInterconnectRemoteLocationsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (InterconnectRemoteLocation) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.interconnectRemoteLocations.get', ordered_params=['project', 'interconnectRemoteLocation'], path_params=['interconnectRemoteLocation', 'project'], query_params=[], relative_path='projects/{project}/global/interconnectRemoteLocations/{interconnectRemoteLocation}', request_field='', request_type_name='ComputeInterconnectRemoteLocationsGetRequest', response_type_name='InterconnectRemoteLocation', supports_download=False)

    def List(self, request, global_params=None):
        """Retrieves the list of interconnect remote locations available to the specified project.

      Args:
        request: (ComputeInterconnectRemoteLocationsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (InterconnectRemoteLocationList) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.interconnectRemoteLocations.list', ordered_params=['project'], path_params=['project'], query_params=['filter', 'maxResults', 'orderBy', 'pageToken', 'returnPartialSuccess'], relative_path='projects/{project}/global/interconnectRemoteLocations', request_field='', request_type_name='ComputeInterconnectRemoteLocationsListRequest', response_type_name='InterconnectRemoteLocationList', supports_download=False)