from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import textwrap
from googlecloudsdk.api_lib.compute.instances.ops_agents import ops_agents_policy as agent_policy
def ConvertOpsAgentPolicyToGuestPolicy(messages, ops_agents_policy, prev_recipes=None):
    """Converts Ops Agent policy to OS Config guest policy."""
    ops_agents_policy_assignment = ops_agents_policy.assignment
    _SetAgentVersion(ops_agents_policy.agent_rules)
    guest_policy = messages.GuestPolicy(description=_CreateDescription(ops_agents_policy.agent_rules, ops_agents_policy.description), etag=ops_agents_policy.etag, assignment=_CreateAssignment(messages, ops_agents_policy_assignment.group_labels, ops_agents_policy_assignment.os_types, ops_agents_policy_assignment.zones, ops_agents_policy_assignment.instances), packages=_CreatePackages(messages, ops_agents_policy.agent_rules, ops_agents_policy_assignment.os_types[0]), packageRepositories=_CreatePackageRepositories(messages, ops_agents_policy_assignment.os_types[0], ops_agents_policy.agent_rules), recipes=_CreateRecipes(messages, ops_agents_policy.agent_rules, ops_agents_policy.assignment.os_types[0], prev_recipes))
    return guest_policy