from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.securitycenter.v1 import securitycenter_v1_messages as messages
class ProjectsEventThreatDetectionSettingsEffectiveCustomModulesService(base_api.BaseApiService):
    """Service class for the projects_eventThreatDetectionSettings_effectiveCustomModules resource."""
    _NAME = 'projects_eventThreatDetectionSettings_effectiveCustomModules'

    def __init__(self, client):
        super(SecuritycenterV1.ProjectsEventThreatDetectionSettingsEffectiveCustomModulesService, self).__init__(client)
        self._upload_configs = {}

    def Get(self, request, global_params=None):
        """Gets an effective Event Threat Detection custom module at the given level.

      Args:
        request: (SecuritycenterProjectsEventThreatDetectionSettingsEffectiveCustomModulesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (EffectiveEventThreatDetectionCustomModule) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/eventThreatDetectionSettings/effectiveCustomModules/{effectiveCustomModulesId}', http_method='GET', method_id='securitycenter.projects.eventThreatDetectionSettings.effectiveCustomModules.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='SecuritycenterProjectsEventThreatDetectionSettingsEffectiveCustomModulesGetRequest', response_type_name='EffectiveEventThreatDetectionCustomModule', supports_download=False)

    def List(self, request, global_params=None):
        """Lists all effective Event Threat Detection custom modules for the given parent. This includes resident modules defined at the scope of the parent along with modules inherited from its ancestors.

      Args:
        request: (SecuritycenterProjectsEventThreatDetectionSettingsEffectiveCustomModulesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListEffectiveEventThreatDetectionCustomModulesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/eventThreatDetectionSettings/effectiveCustomModules', http_method='GET', method_id='securitycenter.projects.eventThreatDetectionSettings.effectiveCustomModules.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v1/{+parent}/effectiveCustomModules', request_field='', request_type_name='SecuritycenterProjectsEventThreatDetectionSettingsEffectiveCustomModulesListRequest', response_type_name='ListEffectiveEventThreatDetectionCustomModulesResponse', supports_download=False)