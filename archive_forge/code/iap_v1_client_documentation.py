from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.iap.v1 import iap_v1_messages as messages
Validates that a given CEL expression conforms to IAP restrictions.

      Args:
        request: (IapValidateAttributeExpressionRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ValidateIapAttributeExpressionResponse) The response message.
      