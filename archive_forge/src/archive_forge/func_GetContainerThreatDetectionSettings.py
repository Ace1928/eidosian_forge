from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.securitycenter.v1beta2 import securitycenter_v1beta2_messages as messages
def GetContainerThreatDetectionSettings(self, request, global_params=None):
    """Get the ContainerThreatDetectionSettings resource. In the returned settings response, a missing field only indicates that it was not explicitly set, so no assumption should be made about these fields. In other words, GetContainerThreatDetectionSettings does not calculate the effective service settings for the resource, which accounts for inherited settings and defaults. Instead, use CalculateContainerThreatDetectionSettings for this purpose.

      Args:
        request: (SecuritycenterProjectsGetContainerThreatDetectionSettingsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ContainerThreatDetectionSettings) The response message.
      """
    config = self.GetMethodConfig('GetContainerThreatDetectionSettings')
    return self._RunMethod(config, request, global_params=global_params)