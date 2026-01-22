from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.monitoring.v3 import monitoring_v3_messages as messages
class ProjectsMetricDescriptorsService(base_api.BaseApiService):
    """Service class for the projects_metricDescriptors resource."""
    _NAME = 'projects_metricDescriptors'

    def __init__(self, client):
        super(MonitoringV3.ProjectsMetricDescriptorsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a new metric descriptor. The creation is executed asynchronously. User-created metric descriptors define custom metrics (https://cloud.google.com/monitoring/custom-metrics). The metric descriptor is updated if it already exists, except that metric labels are never removed.

      Args:
        request: (MetricDescriptor) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (MetricDescriptor) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v3/projects/{projectsId}/metricDescriptors', http_method='POST', method_id='monitoring.projects.metricDescriptors.create', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v3/{+name}/metricDescriptors', request_field='<request>', request_type_name='MetricDescriptor', response_type_name='MetricDescriptor', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a metric descriptor. Only user-created custom metrics (https://cloud.google.com/monitoring/custom-metrics) can be deleted.

      Args:
        request: (MonitoringProjectsMetricDescriptorsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Empty) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v3/projects/{projectsId}/metricDescriptors/{metricDescriptorsId}', http_method='DELETE', method_id='monitoring.projects.metricDescriptors.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v3/{+name}', request_field='', request_type_name='MonitoringProjectsMetricDescriptorsDeleteRequest', response_type_name='Empty', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets a single metric descriptor.

      Args:
        request: (MonitoringProjectsMetricDescriptorsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (MetricDescriptor) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v3/projects/{projectsId}/metricDescriptors/{metricDescriptorsId}', http_method='GET', method_id='monitoring.projects.metricDescriptors.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v3/{+name}', request_field='', request_type_name='MonitoringProjectsMetricDescriptorsGetRequest', response_type_name='MetricDescriptor', supports_download=False)

    def List(self, request, global_params=None):
        """Lists metric descriptors that match a filter.

      Args:
        request: (MonitoringProjectsMetricDescriptorsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListMetricDescriptorsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v3/projects/{projectsId}/metricDescriptors', http_method='GET', method_id='monitoring.projects.metricDescriptors.list', ordered_params=['name'], path_params=['name'], query_params=['filter', 'pageSize', 'pageToken'], relative_path='v3/{+name}/metricDescriptors', request_field='', request_type_name='MonitoringProjectsMetricDescriptorsListRequest', response_type_name='ListMetricDescriptorsResponse', supports_download=False)