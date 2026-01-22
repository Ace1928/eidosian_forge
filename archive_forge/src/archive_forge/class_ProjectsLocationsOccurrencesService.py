from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.containeranalysis.v1 import containeranalysis_v1_messages as messages
class ProjectsLocationsOccurrencesService(base_api.BaseApiService):
    """Service class for the projects_locations_occurrences resource."""
    _NAME = 'projects_locations_occurrences'

    def __init__(self, client):
        super(ContaineranalysisV1.ProjectsLocationsOccurrencesService, self).__init__(client)
        self._upload_configs = {}

    def Get(self, request, global_params=None):
        """Gets the specified occurrence.

      Args:
        request: (ContaineranalysisProjectsLocationsOccurrencesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Occurrence) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/occurrences/{occurrencesId}', http_method='GET', method_id='containeranalysis.projects.locations.occurrences.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='ContaineranalysisProjectsLocationsOccurrencesGetRequest', response_type_name='Occurrence', supports_download=False)

    def GetNotes(self, request, global_params=None):
        """Gets the note attached to the specified occurrence. Consumer projects can use this method to get a note that belongs to a provider project.

      Args:
        request: (ContaineranalysisProjectsLocationsOccurrencesGetNotesRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Note) The response message.
      """
        config = self.GetMethodConfig('GetNotes')
        return self._RunMethod(config, request, global_params=global_params)
    GetNotes.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/occurrences/{occurrencesId}/notes', http_method='GET', method_id='containeranalysis.projects.locations.occurrences.getNotes', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}/notes', request_field='', request_type_name='ContaineranalysisProjectsLocationsOccurrencesGetNotesRequest', response_type_name='Note', supports_download=False)

    def GetVulnerabilitySummary(self, request, global_params=None):
        """Gets a summary of the number and severity of occurrences.

      Args:
        request: (ContaineranalysisProjectsLocationsOccurrencesGetVulnerabilitySummaryRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (VulnerabilityOccurrencesSummary) The response message.
      """
        config = self.GetMethodConfig('GetVulnerabilitySummary')
        return self._RunMethod(config, request, global_params=global_params)
    GetVulnerabilitySummary.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/occurrences:vulnerabilitySummary', http_method='GET', method_id='containeranalysis.projects.locations.occurrences.getVulnerabilitySummary', ordered_params=['parent'], path_params=['parent'], query_params=['filter'], relative_path='v1/{+parent}/occurrences:vulnerabilitySummary', request_field='', request_type_name='ContaineranalysisProjectsLocationsOccurrencesGetVulnerabilitySummaryRequest', response_type_name='VulnerabilityOccurrencesSummary', supports_download=False)

    def List(self, request, global_params=None):
        """Lists occurrences for the specified project.

      Args:
        request: (ContaineranalysisProjectsLocationsOccurrencesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListOccurrencesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/occurrences', http_method='GET', method_id='containeranalysis.projects.locations.occurrences.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'pageSize', 'pageToken'], relative_path='v1/{+parent}/occurrences', request_field='', request_type_name='ContaineranalysisProjectsLocationsOccurrencesListRequest', response_type_name='ListOccurrencesResponse', supports_download=False)