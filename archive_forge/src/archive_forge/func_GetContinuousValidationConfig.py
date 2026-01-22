from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.binaryauthorization.v1alpha2 import binaryauthorization_v1alpha2_messages as messages
def GetContinuousValidationConfig(self, request, global_params=None):
    """Gets the continuous validation config for the project. Returns a default config if the project does not have one. The default config disables continuous validation on all policies.

      Args:
        request: (BinaryauthorizationProjectsGetContinuousValidationConfigRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ContinuousValidationConfig) The response message.
      """
    config = self.GetMethodConfig('GetContinuousValidationConfig')
    return self._RunMethod(config, request, global_params=global_params)