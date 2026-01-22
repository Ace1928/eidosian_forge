from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.securitycentermanagement.v1 import securitycentermanagement_v1_messages as messages
class ProjectsLocationsEffectiveEventThreatDetectionCustomModulesService(base_api.BaseApiService):
    """Service class for the projects_locations_effectiveEventThreatDetectionCustomModules resource."""
    _NAME = 'projects_locations_effectiveEventThreatDetectionCustomModules'

    def __init__(self, client):
        super(SecuritycentermanagementV1.ProjectsLocationsEffectiveEventThreatDetectionCustomModulesService, self).__init__(client)
        self._upload_configs = {}

    def Get(self, request, global_params=None):
        """Gets an effective ETD custom module. Retrieves the effective module at the given level. The difference between an EffectiveCustomModule and a CustomModule is that the fields for an EffectiveCustomModule are computed from ancestors if needed. For example, the enablement_state for a CustomModule can be either ENABLED, DISABLED, or INHERITED. Where as the enablement_state for an EffectiveCustomModule is always computed to ENABLED or DISABLED (the effective enablement_state).

      Args:
        request: (SecuritycentermanagementProjectsLocationsEffectiveEventThreatDetectionCustomModulesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (EffectiveEventThreatDetectionCustomModule) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/effectiveEventThreatDetectionCustomModules/{effectiveEventThreatDetectionCustomModulesId}', http_method='GET', method_id='securitycentermanagement.projects.locations.effectiveEventThreatDetectionCustomModules.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='SecuritycentermanagementProjectsLocationsEffectiveEventThreatDetectionCustomModulesGetRequest', response_type_name='EffectiveEventThreatDetectionCustomModule', supports_download=False)

    def List(self, request, global_params=None):
        """Lists all effective Event Threat Detection custom modules for the given parent. This includes resident modules defined at the scope of the parent along with modules inherited from its ancestors.

      Args:
        request: (SecuritycentermanagementProjectsLocationsEffectiveEventThreatDetectionCustomModulesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListEffectiveEventThreatDetectionCustomModulesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/effectiveEventThreatDetectionCustomModules', http_method='GET', method_id='securitycentermanagement.projects.locations.effectiveEventThreatDetectionCustomModules.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v1/{+parent}/effectiveEventThreatDetectionCustomModules', request_field='', request_type_name='SecuritycentermanagementProjectsLocationsEffectiveEventThreatDetectionCustomModulesListRequest', response_type_name='ListEffectiveEventThreatDetectionCustomModulesResponse', supports_download=False)