from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.cloudbuild.v1 import cloudbuild_v1_messages as messages
class ProjectsLocationsInstallationsService(base_api.BaseApiService):
    """Service class for the projects_locations_installations resource."""
    _NAME = 'projects_locations_installations'

    def __init__(self, client):
        super(CloudbuildV1.ProjectsLocationsInstallationsService, self).__init__(client)
        self._upload_configs = {}

    def Delete(self, request, global_params=None):
        """Delete an association between a GCP project and a GitHub installation. This API is experimental.

      Args:
        request: (CloudbuildProjectsLocationsInstallationsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Empty) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/installations/{installationsId}', http_method='DELETE', method_id='cloudbuild.projects.locations.installations.delete', ordered_params=['name'], path_params=['name'], query_params=['installationId', 'projectId'], relative_path='v1/{+name}', request_field='', request_type_name='CloudbuildProjectsLocationsInstallationsDeleteRequest', response_type_name='Empty', supports_download=False)

    def List(self, request, global_params=None):
        """List all Installations for a given project id. This API is experimental.

      Args:
        request: (CloudbuildProjectsLocationsInstallationsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListGitHubInstallationsForProjectResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/installations', http_method='GET', method_id='cloudbuild.projects.locations.installations.list', ordered_params=['parent'], path_params=['parent'], query_params=['projectId'], relative_path='v1/{+parent}/installations', request_field='', request_type_name='CloudbuildProjectsLocationsInstallationsListRequest', response_type_name='ListGitHubInstallationsForProjectResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Update settings for a GCP project to GitHub installation mapping. This API is experimental.

      Args:
        request: (CloudbuildProjectsLocationsInstallationsPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Empty) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/installations/{installationsId}', http_method='PATCH', method_id='cloudbuild.projects.locations.installations.patch', ordered_params=['name'], path_params=['name'], query_params=['installationId', 'name1', 'projectId', 'updateMask'], relative_path='v1/{+name}', request_field='installation', request_type_name='CloudbuildProjectsLocationsInstallationsPatchRequest', response_type_name='Empty', supports_download=False)