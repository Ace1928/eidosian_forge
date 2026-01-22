from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.monitoring.v3 import monitoring_v3_messages as messages
class ProjectsNotificationChannelDescriptorsService(base_api.BaseApiService):
    """Service class for the projects_notificationChannelDescriptors resource."""
    _NAME = 'projects_notificationChannelDescriptors'

    def __init__(self, client):
        super(MonitoringV3.ProjectsNotificationChannelDescriptorsService, self).__init__(client)
        self._upload_configs = {}

    def Get(self, request, global_params=None):
        """Gets a single channel descriptor. The descriptor indicates which fields are expected / permitted for a notification channel of the given type.

      Args:
        request: (MonitoringProjectsNotificationChannelDescriptorsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (NotificationChannelDescriptor) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v3/projects/{projectsId}/notificationChannelDescriptors/{notificationChannelDescriptorsId}', http_method='GET', method_id='monitoring.projects.notificationChannelDescriptors.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v3/{+name}', request_field='', request_type_name='MonitoringProjectsNotificationChannelDescriptorsGetRequest', response_type_name='NotificationChannelDescriptor', supports_download=False)

    def List(self, request, global_params=None):
        """Lists the descriptors for supported channel types. The use of descriptors makes it possible for new channel types to be dynamically added.

      Args:
        request: (MonitoringProjectsNotificationChannelDescriptorsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListNotificationChannelDescriptorsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v3/projects/{projectsId}/notificationChannelDescriptors', http_method='GET', method_id='monitoring.projects.notificationChannelDescriptors.list', ordered_params=['name'], path_params=['name'], query_params=['pageSize', 'pageToken'], relative_path='v3/{+name}/notificationChannelDescriptors', request_field='', request_type_name='MonitoringProjectsNotificationChannelDescriptorsListRequest', response_type_name='ListNotificationChannelDescriptorsResponse', supports_download=False)