from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.securitycenter.v1beta2 import securitycenter_v1beta2_messages as messages
class ProjectsWebSecurityScannerSettingsService(base_api.BaseApiService):
    """Service class for the projects_webSecurityScannerSettings resource."""
    _NAME = 'projects_webSecurityScannerSettings'

    def __init__(self, client):
        super(SecuritycenterV1beta2.ProjectsWebSecurityScannerSettingsService, self).__init__(client)
        self._upload_configs = {}

    def Calculate(self, request, global_params=None):
        """Calculates the effective WebSecurityScannerSettings based on its level in the resource hierarchy and its settings. Settings provided closer to the target resource take precedence over those further away (e.g. folder will override organization level settings). The default SCC setting for the detector service defaults can be overridden at organization, folder and project levels. No assumptions should be made about the SCC defaults as it is considered an internal implementation detail.

      Args:
        request: (SecuritycenterProjectsWebSecurityScannerSettingsCalculateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (WebSecurityScannerSettings) The response message.
      """
        config = self.GetMethodConfig('Calculate')
        return self._RunMethod(config, request, global_params=global_params)
    Calculate.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta2/projects/{projectsId}/webSecurityScannerSettings:calculate', http_method='GET', method_id='securitycenter.projects.webSecurityScannerSettings.calculate', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1beta2/{+name}:calculate', request_field='', request_type_name='SecuritycenterProjectsWebSecurityScannerSettingsCalculateRequest', response_type_name='WebSecurityScannerSettings', supports_download=False)