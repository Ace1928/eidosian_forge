from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.monitoring.v1 import monitoring_v1_messages as messages
class ProjectsLocationPrometheusApiV1Service(base_api.BaseApiService):
    """Service class for the projects_location_prometheus_api_v1 resource."""
    _NAME = 'projects_location_prometheus_api_v1'

    def __init__(self, client):
        super(MonitoringV1.ProjectsLocationPrometheusApiV1Service, self).__init__(client)
        self._upload_configs = {}

    def Labels(self, request, global_params=None):
        """Lists labels for metrics.

      Args:
        request: (MonitoringProjectsLocationPrometheusApiV1LabelsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (HttpBody) The response message.
      """
        config = self.GetMethodConfig('Labels')
        return self._RunMethod(config, request, global_params=global_params)
    Labels.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/location/{location}/prometheus/api/v1/labels', http_method='POST', method_id='monitoring.projects.location.prometheus.api.v1.labels', ordered_params=['name', 'location'], path_params=['location', 'name'], query_params=[], relative_path='v1/{+name}/location/{location}/prometheus/api/v1/labels', request_field='queryLabelsRequest', request_type_name='MonitoringProjectsLocationPrometheusApiV1LabelsRequest', response_type_name='HttpBody', supports_download=False)

    def Query(self, request, global_params=None):
        """Evaluate a PromQL query at a single point in time.

      Args:
        request: (MonitoringProjectsLocationPrometheusApiV1QueryRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (HttpBody) The response message.
      """
        config = self.GetMethodConfig('Query')
        return self._RunMethod(config, request, global_params=global_params)
    Query.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/location/{location}/prometheus/api/v1/query', http_method='POST', method_id='monitoring.projects.location.prometheus.api.v1.query', ordered_params=['name', 'location'], path_params=['location', 'name'], query_params=[], relative_path='v1/{+name}/location/{location}/prometheus/api/v1/query', request_field='queryInstantRequest', request_type_name='MonitoringProjectsLocationPrometheusApiV1QueryRequest', response_type_name='HttpBody', supports_download=False)

    def QueryExemplars(self, request, global_params=None):
        """Lists exemplars relevant to a given PromQL query,.

      Args:
        request: (MonitoringProjectsLocationPrometheusApiV1QueryExemplarsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (HttpBody) The response message.
      """
        config = self.GetMethodConfig('QueryExemplars')
        return self._RunMethod(config, request, global_params=global_params)
    QueryExemplars.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/location/{location}/prometheus/api/v1/query_exemplars', http_method='POST', method_id='monitoring.projects.location.prometheus.api.v1.query_exemplars', ordered_params=['name', 'location'], path_params=['location', 'name'], query_params=[], relative_path='v1/{+name}/location/{location}/prometheus/api/v1/query_exemplars', request_field='queryExemplarsRequest', request_type_name='MonitoringProjectsLocationPrometheusApiV1QueryExemplarsRequest', response_type_name='HttpBody', supports_download=False)

    def QueryRange(self, request, global_params=None):
        """Evaluate a PromQL query with start, end time range.

      Args:
        request: (MonitoringProjectsLocationPrometheusApiV1QueryRangeRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (HttpBody) The response message.
      """
        config = self.GetMethodConfig('QueryRange')
        return self._RunMethod(config, request, global_params=global_params)
    QueryRange.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/location/{location}/prometheus/api/v1/query_range', http_method='POST', method_id='monitoring.projects.location.prometheus.api.v1.query_range', ordered_params=['name', 'location'], path_params=['location', 'name'], query_params=[], relative_path='v1/{+name}/location/{location}/prometheus/api/v1/query_range', request_field='queryRangeRequest', request_type_name='MonitoringProjectsLocationPrometheusApiV1QueryRangeRequest', response_type_name='HttpBody', supports_download=False)

    def Series(self, request, global_params=None):
        """Lists metadata for metrics.

      Args:
        request: (MonitoringProjectsLocationPrometheusApiV1SeriesRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (HttpBody) The response message.
      """
        config = self.GetMethodConfig('Series')
        return self._RunMethod(config, request, global_params=global_params)
    Series.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/location/{location}/prometheus/api/v1/series', http_method='POST', method_id='monitoring.projects.location.prometheus.api.v1.series', ordered_params=['name', 'location'], path_params=['location', 'name'], query_params=[], relative_path='v1/{+name}/location/{location}/prometheus/api/v1/series', request_field='querySeriesRequest', request_type_name='MonitoringProjectsLocationPrometheusApiV1SeriesRequest', response_type_name='HttpBody', supports_download=False)