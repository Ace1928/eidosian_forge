from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import base
def TroubleshootIAMPolicies(self, access_tuple):
    request = self.messages.GoogleCloudPolicytroubleshooterIamV3TroubleshootIamPolicyRequest(accessTuple=access_tuple)
    return self.client.iam.Troubleshoot(request)