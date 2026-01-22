from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.binaryauthorization.v1alpha2 import binaryauthorization_v1alpha2_messages as messages
def UpdateContinuousValidationConfig(self, request, global_params=None):
    """Updates a project's continuous validation config, and returns a copy of the new config. A config is always updated as a whole to avoid race conditions with concurrent updating requests. Returns NOT_FOUND if the project does not exist, INVALID_ARGUMENT if the request is malformed.

      Args:
        request: (ContinuousValidationConfig) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ContinuousValidationConfig) The response message.
      """
    config = self.GetMethodConfig('UpdateContinuousValidationConfig')
    return self._RunMethod(config, request, global_params=global_params)