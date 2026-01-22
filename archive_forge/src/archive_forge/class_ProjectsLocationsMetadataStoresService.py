from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.aiplatform.v1 import aiplatform_v1_messages as messages
class ProjectsLocationsMetadataStoresService(base_api.BaseApiService):
    """Service class for the projects_locations_metadataStores resource."""
    _NAME = 'projects_locations_metadataStores'

    def __init__(self, client):
        super(AiplatformV1.ProjectsLocationsMetadataStoresService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Initializes a MetadataStore, including allocation of resources.

      Args:
        request: (AiplatformProjectsLocationsMetadataStoresCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/metadataStores', http_method='POST', method_id='aiplatform.projects.locations.metadataStores.create', ordered_params=['parent'], path_params=['parent'], query_params=['metadataStoreId'], relative_path='v1/{+parent}/metadataStores', request_field='googleCloudAiplatformV1MetadataStore', request_type_name='AiplatformProjectsLocationsMetadataStoresCreateRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a single MetadataStore and all its child resources (Artifacts, Executions, and Contexts).

      Args:
        request: (AiplatformProjectsLocationsMetadataStoresDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/metadataStores/{metadataStoresId}', http_method='DELETE', method_id='aiplatform.projects.locations.metadataStores.delete', ordered_params=['name'], path_params=['name'], query_params=['force'], relative_path='v1/{+name}', request_field='', request_type_name='AiplatformProjectsLocationsMetadataStoresDeleteRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def Get(self, request, global_params=None):
        """Retrieves a specific MetadataStore.

      Args:
        request: (AiplatformProjectsLocationsMetadataStoresGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudAiplatformV1MetadataStore) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/metadataStores/{metadataStoresId}', http_method='GET', method_id='aiplatform.projects.locations.metadataStores.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='AiplatformProjectsLocationsMetadataStoresGetRequest', response_type_name='GoogleCloudAiplatformV1MetadataStore', supports_download=False)

    def List(self, request, global_params=None):
        """Lists MetadataStores for a Location.

      Args:
        request: (AiplatformProjectsLocationsMetadataStoresListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudAiplatformV1ListMetadataStoresResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/metadataStores', http_method='GET', method_id='aiplatform.projects.locations.metadataStores.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v1/{+parent}/metadataStores', request_field='', request_type_name='AiplatformProjectsLocationsMetadataStoresListRequest', response_type_name='GoogleCloudAiplatformV1ListMetadataStoresResponse', supports_download=False)