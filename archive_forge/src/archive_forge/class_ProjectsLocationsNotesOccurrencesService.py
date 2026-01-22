from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.containeranalysis.v1 import containeranalysis_v1_messages as messages
class ProjectsLocationsNotesOccurrencesService(base_api.BaseApiService):
    """Service class for the projects_locations_notes_occurrences resource."""
    _NAME = 'projects_locations_notes_occurrences'

    def __init__(self, client):
        super(ContaineranalysisV1.ProjectsLocationsNotesOccurrencesService, self).__init__(client)
        self._upload_configs = {}

    def List(self, request, global_params=None):
        """Lists occurrences referencing the specified note. Provider projects can use this method to get all occurrences across consumer projects referencing the specified note.

      Args:
        request: (ContaineranalysisProjectsLocationsNotesOccurrencesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListNoteOccurrencesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/notes/{notesId}/occurrences', http_method='GET', method_id='containeranalysis.projects.locations.notes.occurrences.list', ordered_params=['name'], path_params=['name'], query_params=['filter', 'pageSize', 'pageToken'], relative_path='v1/{+name}/occurrences', request_field='', request_type_name='ContaineranalysisProjectsLocationsNotesOccurrencesListRequest', response_type_name='ListNoteOccurrencesResponse', supports_download=False)