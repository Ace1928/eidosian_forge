from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.gkemulticloud.v1 import gkemulticloud_v1_messages as messages
class ProjectsLocationsAzureClientsService(base_api.BaseApiService):
    """Service class for the projects_locations_azureClients resource."""
    _NAME = 'projects_locations_azureClients'

    def __init__(self, client):
        super(GkemulticloudV1.ProjectsLocationsAzureClientsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a new AzureClient resource on a given Google Cloud project and region. `AzureClient` resources hold client authentication information needed by the Anthos Multicloud API to manage Azure resources on your Azure subscription on your behalf. If successful, the response contains a newly created Operation resource that can be described to track the status of the operation.

      Args:
        request: (GkemulticloudProjectsLocationsAzureClientsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/azureClients', http_method='POST', method_id='gkemulticloud.projects.locations.azureClients.create', ordered_params=['parent'], path_params=['parent'], query_params=['azureClientId', 'validateOnly'], relative_path='v1/{+parent}/azureClients', request_field='googleCloudGkemulticloudV1AzureClient', request_type_name='GkemulticloudProjectsLocationsAzureClientsCreateRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a specific AzureClient resource. If the client is used by one or more clusters, deletion will fail and a `FAILED_PRECONDITION` error will be returned. If successful, the response contains a newly created Operation resource that can be described to track the status of the operation.

      Args:
        request: (GkemulticloudProjectsLocationsAzureClientsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/azureClients/{azureClientsId}', http_method='DELETE', method_id='gkemulticloud.projects.locations.azureClients.delete', ordered_params=['name'], path_params=['name'], query_params=['allowMissing', 'validateOnly'], relative_path='v1/{+name}', request_field='', request_type_name='GkemulticloudProjectsLocationsAzureClientsDeleteRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def Get(self, request, global_params=None):
        """Describes a specific AzureClient resource.

      Args:
        request: (GkemulticloudProjectsLocationsAzureClientsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudGkemulticloudV1AzureClient) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/azureClients/{azureClientsId}', http_method='GET', method_id='gkemulticloud.projects.locations.azureClients.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='GkemulticloudProjectsLocationsAzureClientsGetRequest', response_type_name='GoogleCloudGkemulticloudV1AzureClient', supports_download=False)

    def List(self, request, global_params=None):
        """Lists all AzureClient resources on a given Google Cloud project and region.

      Args:
        request: (GkemulticloudProjectsLocationsAzureClientsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudGkemulticloudV1ListAzureClientsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/azureClients', http_method='GET', method_id='gkemulticloud.projects.locations.azureClients.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v1/{+parent}/azureClients', request_field='', request_type_name='GkemulticloudProjectsLocationsAzureClientsListRequest', response_type_name='GoogleCloudGkemulticloudV1ListAzureClientsResponse', supports_download=False)