from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.monitoring.v1 import monitoring_v1_messages as messages
class LocationsGlobalMetricsScopesProjectsService(base_api.BaseApiService):
    """Service class for the locations_global_metricsScopes_projects resource."""
    _NAME = 'locations_global_metricsScopes_projects'

    def __init__(self, client):
        super(MonitoringV1.LocationsGlobalMetricsScopesProjectsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Adds a MonitoredProject with the given project ID to the specified Metrics Scope.

      Args:
        request: (MonitoringLocationsGlobalMetricsScopesProjectsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/locations/global/metricsScopes/{metricsScopesId}/projects', http_method='POST', method_id='monitoring.locations.global.metricsScopes.projects.create', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v1/{+parent}/projects', request_field='monitoredProject', request_type_name='MonitoringLocationsGlobalMetricsScopesProjectsCreateRequest', response_type_name='Operation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a MonitoredProject from the specified Metrics Scope.

      Args:
        request: (MonitoringLocationsGlobalMetricsScopesProjectsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/locations/global/metricsScopes/{metricsScopesId}/projects/{projectsId}', http_method='DELETE', method_id='monitoring.locations.global.metricsScopes.projects.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='MonitoringLocationsGlobalMetricsScopesProjectsDeleteRequest', response_type_name='Operation', supports_download=False)