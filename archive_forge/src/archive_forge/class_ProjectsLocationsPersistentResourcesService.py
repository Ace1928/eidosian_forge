from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.aiplatform.v1 import aiplatform_v1_messages as messages
class ProjectsLocationsPersistentResourcesService(base_api.BaseApiService):
    """Service class for the projects_locations_persistentResources resource."""
    _NAME = 'projects_locations_persistentResources'

    def __init__(self, client):
        super(AiplatformV1.ProjectsLocationsPersistentResourcesService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a PersistentResource.

      Args:
        request: (AiplatformProjectsLocationsPersistentResourcesCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/persistentResources', http_method='POST', method_id='aiplatform.projects.locations.persistentResources.create', ordered_params=['parent'], path_params=['parent'], query_params=['persistentResourceId'], relative_path='v1/{+parent}/persistentResources', request_field='googleCloudAiplatformV1PersistentResource', request_type_name='AiplatformProjectsLocationsPersistentResourcesCreateRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a PersistentResource.

      Args:
        request: (AiplatformProjectsLocationsPersistentResourcesDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/persistentResources/{persistentResourcesId}', http_method='DELETE', method_id='aiplatform.projects.locations.persistentResources.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='AiplatformProjectsLocationsPersistentResourcesDeleteRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets a PersistentResource.

      Args:
        request: (AiplatformProjectsLocationsPersistentResourcesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudAiplatformV1PersistentResource) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/persistentResources/{persistentResourcesId}', http_method='GET', method_id='aiplatform.projects.locations.persistentResources.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='AiplatformProjectsLocationsPersistentResourcesGetRequest', response_type_name='GoogleCloudAiplatformV1PersistentResource', supports_download=False)

    def List(self, request, global_params=None):
        """Lists PersistentResources in a Location.

      Args:
        request: (AiplatformProjectsLocationsPersistentResourcesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudAiplatformV1ListPersistentResourcesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/persistentResources', http_method='GET', method_id='aiplatform.projects.locations.persistentResources.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v1/{+parent}/persistentResources', request_field='', request_type_name='AiplatformProjectsLocationsPersistentResourcesListRequest', response_type_name='GoogleCloudAiplatformV1ListPersistentResourcesResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates a PersistentResource.

      Args:
        request: (AiplatformProjectsLocationsPersistentResourcesPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/persistentResources/{persistentResourcesId}', http_method='PATCH', method_id='aiplatform.projects.locations.persistentResources.patch', ordered_params=['name'], path_params=['name'], query_params=['updateMask'], relative_path='v1/{+name}', request_field='googleCloudAiplatformV1PersistentResource', request_type_name='AiplatformProjectsLocationsPersistentResourcesPatchRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def Reboot(self, request, global_params=None):
        """Reboots a PersistentResource.

      Args:
        request: (AiplatformProjectsLocationsPersistentResourcesRebootRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Reboot')
        return self._RunMethod(config, request, global_params=global_params)
    Reboot.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/persistentResources/{persistentResourcesId}:reboot', http_method='POST', method_id='aiplatform.projects.locations.persistentResources.reboot', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}:reboot', request_field='googleCloudAiplatformV1RebootPersistentResourceRequest', request_type_name='AiplatformProjectsLocationsPersistentResourcesRebootRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)