from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.iap.v1 import iap_v1_messages as messages
def ValidateAttributeExpression(self, request, global_params=None):
    """Validates that a given CEL expression conforms to IAP restrictions.

      Args:
        request: (IapValidateAttributeExpressionRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ValidateIapAttributeExpressionResponse) The response message.
      """
    config = self.GetMethodConfig('ValidateAttributeExpression')
    return self._RunMethod(config, request, global_params=global_params)