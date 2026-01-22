from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import utils
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.command_lib.iam import iam_util
def MakeSetEdgeSecurityPolicyRequestTuple(self, security_policy):
    region = getattr(self.ref, 'region', None)
    if region:
        raise calliope_exceptions.InvalidArgumentException('region', 'Can only set edge security policy for global backend services.')
    return (self._client.backendServices, 'SetEdgeSecurityPolicy', self._messages.ComputeBackendServicesSetEdgeSecurityPolicyRequest(securityPolicyReference=self._messages.SecurityPolicyReference(securityPolicy=security_policy), project=self.ref.project, backendService=self.ref.Name()))