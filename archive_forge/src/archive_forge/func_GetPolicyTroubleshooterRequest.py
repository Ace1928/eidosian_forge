from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import base
def GetPolicyTroubleshooterRequest(self, request_time=None):
    return self.messages.GoogleCloudPolicytroubleshooterIamV3ConditionContextRequest(receiveTime=request_time)