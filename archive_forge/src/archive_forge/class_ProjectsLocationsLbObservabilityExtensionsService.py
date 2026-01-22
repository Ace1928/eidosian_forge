from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.networkservices.v1alpha1 import networkservices_v1alpha1_messages as messages
class ProjectsLocationsLbObservabilityExtensionsService(base_api.BaseApiService):
    """Service class for the projects_locations_lbObservabilityExtensions resource."""
    _NAME = 'projects_locations_lbObservabilityExtensions'

    def __init__(self, client):
        super(NetworkservicesV1alpha1.ProjectsLocationsLbObservabilityExtensionsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a new `LbObservabilityExtension` resource in a given project and location.

      Args:
        request: (NetworkservicesProjectsLocationsLbObservabilityExtensionsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/projects/{projectsId}/locations/{locationsId}/lbObservabilityExtensions', http_method='POST', method_id='networkservices.projects.locations.lbObservabilityExtensions.create', ordered_params=['parent'], path_params=['parent'], query_params=['lbObservabilityExtensionId', 'requestId'], relative_path='v1alpha1/{+parent}/lbObservabilityExtensions', request_field='lbObservabilityExtension', request_type_name='NetworkservicesProjectsLocationsLbObservabilityExtensionsCreateRequest', response_type_name='Operation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes the specified `LbObservabilityExtension` resource.

      Args:
        request: (NetworkservicesProjectsLocationsLbObservabilityExtensionsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/projects/{projectsId}/locations/{locationsId}/lbObservabilityExtensions/{lbObservabilityExtensionsId}', http_method='DELETE', method_id='networkservices.projects.locations.lbObservabilityExtensions.delete', ordered_params=['name'], path_params=['name'], query_params=['requestId'], relative_path='v1alpha1/{+name}', request_field='', request_type_name='NetworkservicesProjectsLocationsLbObservabilityExtensionsDeleteRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets details of the specified `LbObservabilityExtension` resource.

      Args:
        request: (NetworkservicesProjectsLocationsLbObservabilityExtensionsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (LbObservabilityExtension) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/projects/{projectsId}/locations/{locationsId}/lbObservabilityExtensions/{lbObservabilityExtensionsId}', http_method='GET', method_id='networkservices.projects.locations.lbObservabilityExtensions.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha1/{+name}', request_field='', request_type_name='NetworkservicesProjectsLocationsLbObservabilityExtensionsGetRequest', response_type_name='LbObservabilityExtension', supports_download=False)

    def List(self, request, global_params=None):
        """Lists `LbObservabilityExtension` resources in a given project and location.

      Args:
        request: (NetworkservicesProjectsLocationsLbObservabilityExtensionsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListLbObservabilityExtensionsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/projects/{projectsId}/locations/{locationsId}/lbObservabilityExtensions', http_method='GET', method_id='networkservices.projects.locations.lbObservabilityExtensions.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'orderBy', 'pageSize', 'pageToken'], relative_path='v1alpha1/{+parent}/lbObservabilityExtensions', request_field='', request_type_name='NetworkservicesProjectsLocationsLbObservabilityExtensionsListRequest', response_type_name='ListLbObservabilityExtensionsResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates the parameters of the specified `LbObservabilityExtension` resource.

      Args:
        request: (NetworkservicesProjectsLocationsLbObservabilityExtensionsPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/projects/{projectsId}/locations/{locationsId}/lbObservabilityExtensions/{lbObservabilityExtensionsId}', http_method='PATCH', method_id='networkservices.projects.locations.lbObservabilityExtensions.patch', ordered_params=['name'], path_params=['name'], query_params=['requestId', 'updateMask'], relative_path='v1alpha1/{+name}', request_field='lbObservabilityExtension', request_type_name='NetworkservicesProjectsLocationsLbObservabilityExtensionsPatchRequest', response_type_name='Operation', supports_download=False)