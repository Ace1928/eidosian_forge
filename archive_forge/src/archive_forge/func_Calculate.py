from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.securitycenter.v1beta2 import securitycenter_v1beta2_messages as messages
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