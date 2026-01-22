from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute.operations import poller
from googlecloudsdk.api_lib.util import waiter
def _MakeListRequestTuple(self, parent_id):
    return (self._client.firewallPolicies, 'List', self._messages.ComputeFirewallPoliciesListRequest(parentId=parent_id))