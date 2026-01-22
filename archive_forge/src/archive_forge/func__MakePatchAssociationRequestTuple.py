from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
def _MakePatchAssociationRequestTuple(self, association, firewall_policy):
    return (self._client.networkFirewallPolicies, 'PatchAssociation', self._messages.ComputeNetworkFirewallPoliciesPatchAssociationRequest(firewallPolicyAssociation=association, firewallPolicy=firewall_policy, project=self.ref.project))