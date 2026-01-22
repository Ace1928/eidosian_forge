from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute.operations import poller
from googlecloudsdk.api_lib.util import waiter
def _MakeCopyRulesRequestTuple(self, dest_sp_id=None, source_security_policy=None):
    return (self._client.organizationSecurityPolicies, 'CopyRules', self._messages.ComputeOrganizationSecurityPoliciesCopyRulesRequest(securityPolicy=dest_sp_id, sourceSecurityPolicy=source_security_policy))