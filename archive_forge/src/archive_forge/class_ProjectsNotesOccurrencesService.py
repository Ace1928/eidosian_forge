from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.containeranalysis.v1 import containeranalysis_v1_messages as messages
class ProjectsNotesOccurrencesService(base_api.BaseApiService):
    """Service class for the projects_notes_occurrences resource."""
    _NAME = 'projects_notes_occurrences'

    def __init__(self, client):
        super(ContaineranalysisV1.ProjectsNotesOccurrencesService, self).__init__(client)
        self._upload_configs = {}

    def List(self, request, global_params=None):
        """Lists occurrences referencing the specified note. Provider projects can use this method to get all occurrences across consumer projects referencing the specified note.

      Args:
        request: (ContaineranalysisProjectsNotesOccurrencesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListNoteOccurrencesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/notes/{notesId}/occurrences', http_method='GET', method_id='containeranalysis.projects.notes.occurrences.list', ordered_params=['name'], path_params=['name'], query_params=['filter', 'pageSize', 'pageToken'], relative_path='v1/{+name}/occurrences', request_field='', request_type_name='ContaineranalysisProjectsNotesOccurrencesListRequest', response_type_name='ListNoteOccurrencesResponse', supports_download=False)