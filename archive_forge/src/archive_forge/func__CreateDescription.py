from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import textwrap
from googlecloudsdk.api_lib.compute.instances.ops_agents import ops_agents_policy as agent_policy
def _CreateDescription(agent_rules, description):
    """Create description in guest policy.

  Args:
    agent_rules: agent rules in ops agent policy.
    description: description in ops agent policy.

  Returns:
    description in guest policy.
  """
    description_template = '{"type": "ops-agents", "description": "%s", "agentRules": [%s]}'
    agent_contents = [agent_rule.ToJson() for agent_rule in agent_rules or []]
    return description_template % (description, ','.join(agent_contents))