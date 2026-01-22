from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.monitoring.v3 import monitoring_v3_messages as messages
class ProjectsSnoozesService(base_api.BaseApiService):
    """Service class for the projects_snoozes resource."""
    _NAME = 'projects_snoozes'

    def __init__(self, client):
        super(MonitoringV3.ProjectsSnoozesService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a Snooze that will prevent alerts, which match the provided criteria, from being opened. The Snooze applies for a specific time interval.

      Args:
        request: (MonitoringProjectsSnoozesCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Snooze) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v3/projects/{projectsId}/snoozes', http_method='POST', method_id='monitoring.projects.snoozes.create', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v3/{+parent}/snoozes', request_field='snooze', request_type_name='MonitoringProjectsSnoozesCreateRequest', response_type_name='Snooze', supports_download=False)

    def Get(self, request, global_params=None):
        """Retrieves a Snooze by name.

      Args:
        request: (MonitoringProjectsSnoozesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Snooze) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v3/projects/{projectsId}/snoozes/{snoozesId}', http_method='GET', method_id='monitoring.projects.snoozes.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v3/{+name}', request_field='', request_type_name='MonitoringProjectsSnoozesGetRequest', response_type_name='Snooze', supports_download=False)

    def List(self, request, global_params=None):
        """Lists the Snoozes associated with a project. Can optionally pass in filter, which specifies predicates to match Snoozes.

      Args:
        request: (MonitoringProjectsSnoozesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListSnoozesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v3/projects/{projectsId}/snoozes', http_method='GET', method_id='monitoring.projects.snoozes.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'pageSize', 'pageToken'], relative_path='v3/{+parent}/snoozes', request_field='', request_type_name='MonitoringProjectsSnoozesListRequest', response_type_name='ListSnoozesResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates a Snooze, identified by its name, with the parameters in the given Snooze object.

      Args:
        request: (MonitoringProjectsSnoozesPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Snooze) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v3/projects/{projectsId}/snoozes/{snoozesId}', http_method='PATCH', method_id='monitoring.projects.snoozes.patch', ordered_params=['name'], path_params=['name'], query_params=['updateMask'], relative_path='v3/{+name}', request_field='snooze', request_type_name='MonitoringProjectsSnoozesPatchRequest', response_type_name='Snooze', supports_download=False)