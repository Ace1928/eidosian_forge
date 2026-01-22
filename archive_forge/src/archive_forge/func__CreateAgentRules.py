from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
from googlecloudsdk.api_lib.compute.instances.ops_agents import ops_agents_policy as agent_policy
from googlecloudsdk.calliope import exceptions
def _CreateAgentRules(agent_rules):
    """Create agent rules in ops agent policy.

  Args:
    agent_rules: json objects.

  Returns:
    agent rules in ops agent policy.
  """
    ops_agent_rules = []
    for agent_rule in agent_rules or []:
        try:
            ops_agent_rules.append(agent_policy.OpsAgentPolicy.AgentRule(agent_rule['type'], agent_rule['enableAutoupgrade'], agent_rule['version'], agent_rule['packageState']))
        except KeyError as e:
            raise exceptions.BadArgumentException('description.agentRules', 'agent rule specification %s missing a required key: %s' % (agent_rule, e))
    return ops_agent_rules