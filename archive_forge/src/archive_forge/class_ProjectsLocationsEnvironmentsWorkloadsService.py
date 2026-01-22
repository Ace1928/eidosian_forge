from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.composer.v1alpha2 import composer_v1alpha2_messages as messages
class ProjectsLocationsEnvironmentsWorkloadsService(base_api.BaseApiService):
    """Service class for the projects_locations_environments_workloads resource."""
    _NAME = 'projects_locations_environments_workloads'

    def __init__(self, client):
        super(ComposerV1alpha2.ProjectsLocationsEnvironmentsWorkloadsService, self).__init__(client)
        self._upload_configs = {}

    def List(self, request, global_params=None):
        """Lists workloads in a Cloud Composer environment. Workload is a unit that runs a single Composer component. This method is supported for Cloud Composer environments in versions composer-3.*.*-airflow-*.*.* and newer.

      Args:
        request: (ComposerProjectsLocationsEnvironmentsWorkloadsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListWorkloadsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha2/projects/{projectsId}/locations/{locationsId}/environments/{environmentsId}/workloads', http_method='GET', method_id='composer.projects.locations.environments.workloads.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'pageSize', 'pageToken'], relative_path='v1alpha2/{+parent}/workloads', request_field='', request_type_name='ComposerProjectsLocationsEnvironmentsWorkloadsListRequest', response_type_name='ListWorkloadsResponse', supports_download=False)