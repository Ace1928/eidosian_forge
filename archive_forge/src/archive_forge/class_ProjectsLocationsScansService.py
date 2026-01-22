from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.ondemandscanning.v1 import ondemandscanning_v1_messages as messages
class ProjectsLocationsScansService(base_api.BaseApiService):
    """Service class for the projects_locations_scans resource."""
    _NAME = 'projects_locations_scans'

    def __init__(self, client):
        super(OndemandscanningV1.ProjectsLocationsScansService, self).__init__(client)
        self._upload_configs = {}

    def AnalyzePackages(self, request, global_params=None):
        """Initiates an analysis of the provided packages.

      Args:
        request: (OndemandscanningProjectsLocationsScansAnalyzePackagesRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('AnalyzePackages')
        return self._RunMethod(config, request, global_params=global_params)
    AnalyzePackages.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/scans:analyzePackages', http_method='POST', method_id='ondemandscanning.projects.locations.scans.analyzePackages', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v1/{+parent}/scans:analyzePackages', request_field='analyzePackagesRequestV1', request_type_name='OndemandscanningProjectsLocationsScansAnalyzePackagesRequest', response_type_name='Operation', supports_download=False)