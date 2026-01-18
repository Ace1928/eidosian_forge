from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.policytroubleshooter.v2alpha1 import policytroubleshooter_v2alpha1_messages as messages
Checks whether a member has a specific permission for a specific resource,.
and explains why the member does or does not have that permission.

      Args:
        request: (GoogleCloudPolicytroubleshooterV2alpha1TroubleshootIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudPolicytroubleshooterV2alpha1TroubleshootIamPolicyResponse) The response message.
      