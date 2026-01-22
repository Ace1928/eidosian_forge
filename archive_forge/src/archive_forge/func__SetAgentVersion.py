from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import textwrap
from googlecloudsdk.api_lib.compute.instances.ops_agents import ops_agents_policy as agent_policy
def _SetAgentVersion(agent_rules):
    for agent_rule in agent_rules or []:
        if agent_rule.version in {'current-major', None, ''}:
            agent_rule.version = _AGENT_RULE_TEMPLATES[agent_rule.type].current_major_version