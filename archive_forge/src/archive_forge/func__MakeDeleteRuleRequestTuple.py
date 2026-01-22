from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute.operations import poller
from googlecloudsdk.api_lib.util import waiter
def _MakeDeleteRuleRequestTuple(self, priority=None, firewall_policy=None):
    return (self._client.firewallPolicies, 'RemoveRule', self._messages.ComputeFirewallPoliciesRemoveRuleRequest(firewallPolicy=firewall_policy, priority=priority))