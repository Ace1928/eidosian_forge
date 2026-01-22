from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.securitycenter.v1beta2 import securitycenter_v1beta2_messages as messages
def GetEventThreatDetectionSettings(self, request, global_params=None):
    """Get the EventThreatDetectionSettings resource. In the returned settings response, a missing field only indicates that it was not explicitly set, so no assumption should be made about these fields. In other words, GetEventThreatDetectionSettings does not calculate the effective service settings for the resource, which accounts for inherited settings and defaults. Instead, use CalculateEventThreatDetectionSettings for this purpose.

      Args:
        request: (SecuritycenterProjectsGetEventThreatDetectionSettingsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (EventThreatDetectionSettings) The response message.
      """
    config = self.GetMethodConfig('GetEventThreatDetectionSettings')
    return self._RunMethod(config, request, global_params=global_params)