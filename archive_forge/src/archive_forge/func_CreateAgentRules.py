from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import enum
import json
import sys
from googlecloudsdk.core.resource import resource_property
def CreateAgentRules(agent_rules):
    """Create agent rules in ops agent policy.

  Args:
    agent_rules: list of dict, fields describing agent rules from the command
      line.

  Returns:
    An OpsAgentPolicy.AgentRules object.
  """
    ops_agents = []
    for agent_rule in agent_rules or []:
        ops_agents.append(OpsAgentPolicy.AgentRule(OpsAgentPolicy.AgentRule.Type(agent_rule['type']), agent_rule['enable-autoupgrade'], agent_rule.get('version', OpsAgentPolicy.AgentRule.Version.CURRENT_MAJOR), OpsAgentPolicy.AgentRule.PackageState(agent_rule.get('package-state', OpsAgentPolicy.AgentRule.PackageState.INSTALLED))))
    return ops_agents