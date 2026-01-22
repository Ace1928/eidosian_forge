from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import base
def GetPolicyTroubleshooterResource(self, resource_name=None, resource_service=None, resource_type=None):
    return self.messages.GoogleCloudPolicytroubleshooterIamV3ConditionContextResource(name=resource_name, service=resource_service, type=resource_type)