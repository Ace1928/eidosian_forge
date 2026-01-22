from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute.operations import poller
from googlecloudsdk.api_lib.util import waiter
def _MakeCloneRulesRequestTuple(self, dest_fp_id=None, source_firewall_policy=None):
    return (self._client.firewallPolicies, 'CloneRules', self._messages.ComputeFirewallPoliciesCloneRulesRequest(firewallPolicy=dest_fp_id, sourceFirewallPolicy=source_firewall_policy))