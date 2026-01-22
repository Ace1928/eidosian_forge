from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.bigquerydatatransfer.v1 import bigquerydatatransfer_v1_messages as messages
class ProjectsLocationsTransferConfigsRunsTransferLogsService(base_api.BaseApiService):
    """Service class for the projects_locations_transferConfigs_runs_transferLogs resource."""
    _NAME = 'projects_locations_transferConfigs_runs_transferLogs'

    def __init__(self, client):
        super(BigquerydatatransferV1.ProjectsLocationsTransferConfigsRunsTransferLogsService, self).__init__(client)
        self._upload_configs = {}

    def List(self, request, global_params=None):
        """Returns log messages for the transfer run.

      Args:
        request: (BigquerydatatransferProjectsLocationsTransferConfigsRunsTransferLogsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListTransferLogsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/transferConfigs/{transferConfigsId}/runs/{runsId}/transferLogs', http_method='GET', method_id='bigquerydatatransfer.projects.locations.transferConfigs.runs.transferLogs.list', ordered_params=['parent'], path_params=['parent'], query_params=['messageTypes', 'pageSize', 'pageToken'], relative_path='v1/{+parent}/transferLogs', request_field='', request_type_name='BigquerydatatransferProjectsLocationsTransferConfigsRunsTransferLogsListRequest', response_type_name='ListTransferLogsResponse', supports_download=False)