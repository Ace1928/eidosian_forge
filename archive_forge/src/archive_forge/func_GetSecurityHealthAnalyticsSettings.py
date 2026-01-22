from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.securitycenter.v1beta2 import securitycenter_v1beta2_messages as messages
def GetSecurityHealthAnalyticsSettings(self, request, global_params=None):
    """Get the SecurityHealthAnalyticsSettings resource. In the returned settings response, a missing field only indicates that it was not explicitly set, so no assumption should be made about these fields. In other words, GetSecurityHealthAnalyticsSettings does not calculate the effective service settings for the resource, which accounts for inherited settings and defaults. Instead, use CalculateSecurityHealthAnalyticsSettings for this purpose.

      Args:
        request: (SecuritycenterProjectsGetSecurityHealthAnalyticsSettingsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (SecurityHealthAnalyticsSettings) The response message.
      """
    config = self.GetMethodConfig('GetSecurityHealthAnalyticsSettings')
    return self._RunMethod(config, request, global_params=global_params)