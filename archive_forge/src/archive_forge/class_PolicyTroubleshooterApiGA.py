from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import base
class PolicyTroubleshooterApiGA(PolicyTroubleshooterApi):
    """Base Class for Policy Troubleshooter API GA."""

    def TroubleshootIAMPolicies(self, access_tuple):
        request = self.messages.GoogleCloudPolicytroubleshooterIamV3TroubleshootIamPolicyRequest(accessTuple=access_tuple)
        return self.client.iam.Troubleshoot(request)

    def GetPolicyTroubleshooterAccessTuple(self, condition_context=None, full_resource_name=None, principal_email=None, permission=None):
        return self.messages.GoogleCloudPolicytroubleshooterIamV3AccessTuple(fullResourceName=full_resource_name, principal=principal_email, permission=permission, conditionContext=condition_context)

    def GetPolicyTroubleshooterRequest(self, request_time=None):
        return self.messages.GoogleCloudPolicytroubleshooterIamV3ConditionContextRequest(receiveTime=request_time)

    def GetPolicyTroubleshooterResource(self, resource_name=None, resource_service=None, resource_type=None):
        return self.messages.GoogleCloudPolicytroubleshooterIamV3ConditionContextResource(name=resource_name, service=resource_service, type=resource_type)

    def GetPolicyTroubleshooterPeer(self, destination_ip=None, destination_port=None):
        return self.messages.GoogleCloudPolicytroubleshooterIamV3ConditionContextPeer(ip=destination_ip, port=destination_port)

    def GetPolicyTroubleshooterConditionContext(self, destination=None, request=None, resource=None):
        return self.messages.GoogleCloudPolicytroubleshooterIamV3ConditionContext(destination=destination, request=request, resource=resource)