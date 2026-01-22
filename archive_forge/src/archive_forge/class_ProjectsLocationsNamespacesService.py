from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.run.v1 import run_v1_messages as messages
class ProjectsLocationsNamespacesService(base_api.BaseApiService):
    """Service class for the projects_locations_namespaces resource."""
    _NAME = 'projects_locations_namespaces'

    def __init__(self, client):
        super(RunV1.ProjectsLocationsNamespacesService, self).__init__(client)
        self._upload_configs = {}

    def Get(self, request, global_params=None):
        """Rpc to get information about a namespace.

      Args:
        request: (RunProjectsLocationsNamespacesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Namespace) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/namespaces/{namespacesId}', http_method='GET', method_id='run.projects.locations.namespaces.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='RunProjectsLocationsNamespacesGetRequest', response_type_name='Namespace', supports_download=False)

    def Patch(self, request, global_params=None):
        """Rpc to update a namespace.

      Args:
        request: (RunProjectsLocationsNamespacesPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Namespace) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/namespaces/{namespacesId}', http_method='PATCH', method_id='run.projects.locations.namespaces.patch', ordered_params=['name'], path_params=['name'], query_params=['updateMask'], relative_path='v1/{+name}', request_field='namespace', request_type_name='RunProjectsLocationsNamespacesPatchRequest', response_type_name='Namespace', supports_download=False)