from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.aiplatform.v1beta1 import aiplatform_v1beta1_messages as messages
class ProjectsLocationsExtensionsService(base_api.BaseApiService):
    """Service class for the projects_locations_extensions resource."""
    _NAME = 'projects_locations_extensions'

    def __init__(self, client):
        super(AiplatformV1beta1.ProjectsLocationsExtensionsService, self).__init__(client)
        self._upload_configs = {}

    def Delete(self, request, global_params=None):
        """Deletes an Extension.

      Args:
        request: (AiplatformProjectsLocationsExtensionsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta1/projects/{projectsId}/locations/{locationsId}/extensions/{extensionsId}', http_method='DELETE', method_id='aiplatform.projects.locations.extensions.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1beta1/{+name}', request_field='', request_type_name='AiplatformProjectsLocationsExtensionsDeleteRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def Execute(self, request, global_params=None):
        """Executes the request against a given extension.

      Args:
        request: (AiplatformProjectsLocationsExtensionsExecuteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudAiplatformV1beta1ExecuteExtensionResponse) The response message.
      """
        config = self.GetMethodConfig('Execute')
        return self._RunMethod(config, request, global_params=global_params)
    Execute.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta1/projects/{projectsId}/locations/{locationsId}/extensions/{extensionsId}:execute', http_method='POST', method_id='aiplatform.projects.locations.extensions.execute', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1beta1/{+name}:execute', request_field='googleCloudAiplatformV1beta1ExecuteExtensionRequest', request_type_name='AiplatformProjectsLocationsExtensionsExecuteRequest', response_type_name='GoogleCloudAiplatformV1beta1ExecuteExtensionResponse', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets an Extension.

      Args:
        request: (AiplatformProjectsLocationsExtensionsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudAiplatformV1beta1Extension) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta1/projects/{projectsId}/locations/{locationsId}/extensions/{extensionsId}', http_method='GET', method_id='aiplatform.projects.locations.extensions.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1beta1/{+name}', request_field='', request_type_name='AiplatformProjectsLocationsExtensionsGetRequest', response_type_name='GoogleCloudAiplatformV1beta1Extension', supports_download=False)

    def Import(self, request, global_params=None):
        """Imports an Extension.

      Args:
        request: (AiplatformProjectsLocationsExtensionsImportRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Import')
        return self._RunMethod(config, request, global_params=global_params)
    Import.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta1/projects/{projectsId}/locations/{locationsId}/extensions:import', http_method='POST', method_id='aiplatform.projects.locations.extensions.import', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v1beta1/{+parent}/extensions:import', request_field='googleCloudAiplatformV1beta1Extension', request_type_name='AiplatformProjectsLocationsExtensionsImportRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def List(self, request, global_params=None):
        """Lists Extensions in a location.

      Args:
        request: (AiplatformProjectsLocationsExtensionsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudAiplatformV1beta1ListExtensionsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta1/projects/{projectsId}/locations/{locationsId}/extensions', http_method='GET', method_id='aiplatform.projects.locations.extensions.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'orderBy', 'pageSize', 'pageToken'], relative_path='v1beta1/{+parent}/extensions', request_field='', request_type_name='AiplatformProjectsLocationsExtensionsListRequest', response_type_name='GoogleCloudAiplatformV1beta1ListExtensionsResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates an Extension.

      Args:
        request: (AiplatformProjectsLocationsExtensionsPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudAiplatformV1beta1Extension) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta1/projects/{projectsId}/locations/{locationsId}/extensions/{extensionsId}', http_method='PATCH', method_id='aiplatform.projects.locations.extensions.patch', ordered_params=['name'], path_params=['name'], query_params=['updateMask'], relative_path='v1beta1/{+name}', request_field='googleCloudAiplatformV1beta1Extension', request_type_name='AiplatformProjectsLocationsExtensionsPatchRequest', response_type_name='GoogleCloudAiplatformV1beta1Extension', supports_download=False)

    def Query(self, request, global_params=None):
        """Queries an extension with a default controller.

      Args:
        request: (AiplatformProjectsLocationsExtensionsQueryRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudAiplatformV1beta1QueryExtensionResponse) The response message.
      """
        config = self.GetMethodConfig('Query')
        return self._RunMethod(config, request, global_params=global_params)
    Query.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta1/projects/{projectsId}/locations/{locationsId}/extensions/{extensionsId}:query', http_method='POST', method_id='aiplatform.projects.locations.extensions.query', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1beta1/{+name}:query', request_field='googleCloudAiplatformV1beta1QueryExtensionRequest', request_type_name='AiplatformProjectsLocationsExtensionsQueryRequest', response_type_name='GoogleCloudAiplatformV1beta1QueryExtensionResponse', supports_download=False)