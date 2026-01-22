from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.securitycenter.v1 import securitycenter_v1_messages as messages
def ValidateCustomModule(self, request, global_params=None):
    """Validates the given Event Threat Detection custom module.

      Args:
        request: (SecuritycenterProjectsEventThreatDetectionSettingsValidateCustomModuleRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ValidateEventThreatDetectionCustomModuleResponse) The response message.
      """
    config = self.GetMethodConfig('ValidateCustomModule')
    return self._RunMethod(config, request, global_params=global_params)