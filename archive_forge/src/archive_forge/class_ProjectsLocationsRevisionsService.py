from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.run.v1 import run_v1_messages as messages
class ProjectsLocationsRevisionsService(base_api.BaseApiService):
    """Service class for the projects_locations_revisions resource."""
    _NAME = 'projects_locations_revisions'

    def __init__(self, client):
        super(RunV1.ProjectsLocationsRevisionsService, self).__init__(client)
        self._upload_configs = {}

    def Delete(self, request, global_params=None):
        """Delete a revision.

      Args:
        request: (RunProjectsLocationsRevisionsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Status) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/revisions/{revisionsId}', http_method='DELETE', method_id='run.projects.locations.revisions.delete', ordered_params=['name'], path_params=['name'], query_params=['apiVersion', 'dryRun', 'kind', 'propagationPolicy'], relative_path='v1/{+name}', request_field='', request_type_name='RunProjectsLocationsRevisionsDeleteRequest', response_type_name='Status', supports_download=False)

    def Get(self, request, global_params=None):
        """Get information about a revision.

      Args:
        request: (RunProjectsLocationsRevisionsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Revision) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/revisions/{revisionsId}', http_method='GET', method_id='run.projects.locations.revisions.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='RunProjectsLocationsRevisionsGetRequest', response_type_name='Revision', supports_download=False)

    def List(self, request, global_params=None):
        """List revisions.

      Args:
        request: (RunProjectsLocationsRevisionsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListRevisionsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/revisions', http_method='GET', method_id='run.projects.locations.revisions.list', ordered_params=['parent'], path_params=['parent'], query_params=['continue_', 'fieldSelector', 'includeUninitialized', 'labelSelector', 'limit', 'resourceVersion', 'watch'], relative_path='v1/{+parent}/revisions', request_field='', request_type_name='RunProjectsLocationsRevisionsListRequest', response_type_name='ListRevisionsResponse', supports_download=False)