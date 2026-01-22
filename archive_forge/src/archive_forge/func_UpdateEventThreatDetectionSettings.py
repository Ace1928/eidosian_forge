from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.securitycenter.v1beta2 import securitycenter_v1beta2_messages as messages
def UpdateEventThreatDetectionSettings(self, request, global_params=None):
    """Update the EventThreatDetectionSettings resource.

      Args:
        request: (SecuritycenterProjectsUpdateEventThreatDetectionSettingsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (EventThreatDetectionSettings) The response message.
      """
    config = self.GetMethodConfig('UpdateEventThreatDetectionSettings')
    return self._RunMethod(config, request, global_params=global_params)