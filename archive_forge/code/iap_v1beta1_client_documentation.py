from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.iap.v1beta1 import iap_v1beta1_messages as messages
Returns permissions that a caller has on the Identity-Aware Proxy protected resource. If the resource does not exist or the caller does not have Identity-Aware Proxy permissions a [google.rpc.Code.PERMISSION_DENIED] will be returned. More information about managing access via IAP can be found at: https://cloud.google.com/iap/docs/managing-access#managing_access_via_the_api.

      Args:
        request: (IapTestIamPermissionsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (TestIamPermissionsResponse) The response message.
      