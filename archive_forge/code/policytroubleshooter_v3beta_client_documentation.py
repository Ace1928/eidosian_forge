from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.policytroubleshooter.v3beta import policytroubleshooter_v3beta_messages as messages
Checks whether a principal has a specific permission for a specific resource, and explains why the principal does or doesn't have that permission.

      Args:
        request: (GoogleCloudPolicytroubleshooterIamV3betaTroubleshootIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudPolicytroubleshooterIamV3betaTroubleshootIamPolicyResponse) The response message.
      