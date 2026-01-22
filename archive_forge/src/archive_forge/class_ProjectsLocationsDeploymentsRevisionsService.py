from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.config.v1alpha2 import config_v1alpha2_messages as messages
class ProjectsLocationsDeploymentsRevisionsService(base_api.BaseApiService):
    """Service class for the projects_locations_deployments_revisions resource."""
    _NAME = 'projects_locations_deployments_revisions'

    def __init__(self, client):
        super(ConfigV1alpha2.ProjectsLocationsDeploymentsRevisionsService, self).__init__(client)
        self._upload_configs = {}

    def ExportState(self, request, global_params=None):
        """Exports Terraform state file from a given revision.

      Args:
        request: (ConfigProjectsLocationsDeploymentsRevisionsExportStateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Statefile) The response message.
      """
        config = self.GetMethodConfig('ExportState')
        return self._RunMethod(config, request, global_params=global_params)
    ExportState.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha2/projects/{projectsId}/locations/{locationsId}/deployments/{deploymentsId}/revisions/{revisionsId}:exportState', http_method='POST', method_id='config.projects.locations.deployments.revisions.exportState', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v1alpha2/{+parent}:exportState', request_field='exportRevisionStatefileRequest', request_type_name='ConfigProjectsLocationsDeploymentsRevisionsExportStateRequest', response_type_name='Statefile', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets details about a Revision.

      Args:
        request: (ConfigProjectsLocationsDeploymentsRevisionsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Revision) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha2/projects/{projectsId}/locations/{locationsId}/deployments/{deploymentsId}/revisions/{revisionsId}', http_method='GET', method_id='config.projects.locations.deployments.revisions.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha2/{+name}', request_field='', request_type_name='ConfigProjectsLocationsDeploymentsRevisionsGetRequest', response_type_name='Revision', supports_download=False)

    def List(self, request, global_params=None):
        """Lists Revisions of a deployment.

      Args:
        request: (ConfigProjectsLocationsDeploymentsRevisionsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListRevisionsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha2/projects/{projectsId}/locations/{locationsId}/deployments/{deploymentsId}/revisions', http_method='GET', method_id='config.projects.locations.deployments.revisions.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'orderBy', 'pageSize', 'pageToken'], relative_path='v1alpha2/{+parent}/revisions', request_field='', request_type_name='ConfigProjectsLocationsDeploymentsRevisionsListRequest', response_type_name='ListRevisionsResponse', supports_download=False)