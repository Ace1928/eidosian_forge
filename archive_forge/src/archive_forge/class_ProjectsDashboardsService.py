from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.monitoring.v1 import monitoring_v1_messages as messages
class ProjectsDashboardsService(base_api.BaseApiService):
    """Service class for the projects_dashboards resource."""
    _NAME = 'projects_dashboards'

    def __init__(self, client):
        super(MonitoringV1.ProjectsDashboardsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a new custom dashboard. For examples on how you can use this API to create dashboards, see Managing dashboards by API (https://cloud.google.com/monitoring/dashboards/api-dashboard). This method requires the monitoring.dashboards.create permission on the specified project. For more information about permissions, see Cloud Identity and Access Management (https://cloud.google.com/iam).

      Args:
        request: (MonitoringProjectsDashboardsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Dashboard) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/dashboards', http_method='POST', method_id='monitoring.projects.dashboards.create', ordered_params=['parent'], path_params=['parent'], query_params=['validateOnly'], relative_path='v1/{+parent}/dashboards', request_field='dashboard', request_type_name='MonitoringProjectsDashboardsCreateRequest', response_type_name='Dashboard', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes an existing custom dashboard.This method requires the monitoring.dashboards.delete permission on the specified dashboard. For more information, see Cloud Identity and Access Management (https://cloud.google.com/iam).

      Args:
        request: (MonitoringProjectsDashboardsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Empty) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/dashboards/{dashboardsId}', http_method='DELETE', method_id='monitoring.projects.dashboards.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='MonitoringProjectsDashboardsDeleteRequest', response_type_name='Empty', supports_download=False)

    def Get(self, request, global_params=None):
        """Fetches a specific dashboard.This method requires the monitoring.dashboards.get permission on the specified dashboard. For more information, see Cloud Identity and Access Management (https://cloud.google.com/iam).

      Args:
        request: (MonitoringProjectsDashboardsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Dashboard) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/dashboards/{dashboardsId}', http_method='GET', method_id='monitoring.projects.dashboards.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='MonitoringProjectsDashboardsGetRequest', response_type_name='Dashboard', supports_download=False)

    def List(self, request, global_params=None):
        """Lists the existing dashboards.This method requires the monitoring.dashboards.list permission on the specified project. For more information, see Cloud Identity and Access Management (https://cloud.google.com/iam).

      Args:
        request: (MonitoringProjectsDashboardsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListDashboardsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/dashboards', http_method='GET', method_id='monitoring.projects.dashboards.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v1/{+parent}/dashboards', request_field='', request_type_name='MonitoringProjectsDashboardsListRequest', response_type_name='ListDashboardsResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Replaces an existing custom dashboard with a new definition.This method requires the monitoring.dashboards.update permission on the specified dashboard. For more information, see Cloud Identity and Access Management (https://cloud.google.com/iam).

      Args:
        request: (MonitoringProjectsDashboardsPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Dashboard) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/dashboards/{dashboardsId}', http_method='PATCH', method_id='monitoring.projects.dashboards.patch', ordered_params=['name'], path_params=['name'], query_params=['validateOnly'], relative_path='v1/{+name}', request_field='dashboard', request_type_name='MonitoringProjectsDashboardsPatchRequest', response_type_name='Dashboard', supports_download=False)