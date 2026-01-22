from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.securitycenter.v1beta2 import securitycenter_v1beta2_messages as messages
def UpdateVirtualMachineThreatDetectionSettings(self, request, global_params=None):
    """Update the VirtualMachineThreatDetectionSettings resource.

      Args:
        request: (SecuritycenterProjectsUpdateVirtualMachineThreatDetectionSettingsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (VirtualMachineThreatDetectionSettings) The response message.
      """
    config = self.GetMethodConfig('UpdateVirtualMachineThreatDetectionSettings')
    return self._RunMethod(config, request, global_params=global_params)