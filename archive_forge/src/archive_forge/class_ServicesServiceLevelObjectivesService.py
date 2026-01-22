from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.monitoring.v3 import monitoring_v3_messages as messages
class ServicesServiceLevelObjectivesService(base_api.BaseApiService):
    """Service class for the services_serviceLevelObjectives resource."""
    _NAME = 'services_serviceLevelObjectives'

    def __init__(self, client):
        super(MonitoringV3.ServicesServiceLevelObjectivesService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Create a ServiceLevelObjective for the given Service.

      Args:
        request: (MonitoringServicesServiceLevelObjectivesCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ServiceLevelObjective) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v3/{v3Id}/{v3Id1}/services/{servicesId}/serviceLevelObjectives', http_method='POST', method_id='monitoring.services.serviceLevelObjectives.create', ordered_params=['parent'], path_params=['parent'], query_params=['serviceLevelObjectiveId'], relative_path='v3/{+parent}/serviceLevelObjectives', request_field='serviceLevelObjective', request_type_name='MonitoringServicesServiceLevelObjectivesCreateRequest', response_type_name='ServiceLevelObjective', supports_download=False)

    def Delete(self, request, global_params=None):
        """Delete the given ServiceLevelObjective.

      Args:
        request: (MonitoringServicesServiceLevelObjectivesDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Empty) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v3/{v3Id}/{v3Id1}/services/{servicesId}/serviceLevelObjectives/{serviceLevelObjectivesId}', http_method='DELETE', method_id='monitoring.services.serviceLevelObjectives.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v3/{+name}', request_field='', request_type_name='MonitoringServicesServiceLevelObjectivesDeleteRequest', response_type_name='Empty', supports_download=False)

    def Get(self, request, global_params=None):
        """Get a ServiceLevelObjective by name.

      Args:
        request: (MonitoringServicesServiceLevelObjectivesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ServiceLevelObjective) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v3/{v3Id}/{v3Id1}/services/{servicesId}/serviceLevelObjectives/{serviceLevelObjectivesId}', http_method='GET', method_id='monitoring.services.serviceLevelObjectives.get', ordered_params=['name'], path_params=['name'], query_params=['view'], relative_path='v3/{+name}', request_field='', request_type_name='MonitoringServicesServiceLevelObjectivesGetRequest', response_type_name='ServiceLevelObjective', supports_download=False)

    def List(self, request, global_params=None):
        """List the ServiceLevelObjectives for the given Service.

      Args:
        request: (MonitoringServicesServiceLevelObjectivesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListServiceLevelObjectivesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v3/{v3Id}/{v3Id1}/services/{servicesId}/serviceLevelObjectives', http_method='GET', method_id='monitoring.services.serviceLevelObjectives.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'pageSize', 'pageToken', 'view'], relative_path='v3/{+parent}/serviceLevelObjectives', request_field='', request_type_name='MonitoringServicesServiceLevelObjectivesListRequest', response_type_name='ListServiceLevelObjectivesResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Update the given ServiceLevelObjective.

      Args:
        request: (MonitoringServicesServiceLevelObjectivesPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ServiceLevelObjective) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v3/{v3Id}/{v3Id1}/services/{servicesId}/serviceLevelObjectives/{serviceLevelObjectivesId}', http_method='PATCH', method_id='monitoring.services.serviceLevelObjectives.patch', ordered_params=['name'], path_params=['name'], query_params=['updateMask'], relative_path='v3/{+name}', request_field='serviceLevelObjective', request_type_name='MonitoringServicesServiceLevelObjectivesPatchRequest', response_type_name='ServiceLevelObjective', supports_download=False)