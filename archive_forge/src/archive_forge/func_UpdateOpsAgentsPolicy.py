from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import enum
import json
import sys
from googlecloudsdk.core.resource import resource_property
def UpdateOpsAgentsPolicy(ops_agents_policy, description, etag, agent_rules, os_types, group_labels, zones, instances):
    """Merge existing ops agent policy with user updates.

  Unless explicitly mentioned, a None value means "leave unchanged".

  Args:
    ops_agents_policy: OpsAgentPolicy, ops agent policy.
    description: str, ops agent policy description, or None.
    etag: str, unique tag for policy to prevent race conditions, or None.
    agent_rules: list of dict, fields describing agent rules from the command
      line, or None. An empty list means the same as None.
    os_types: dict, VM OS type matchers, or None.
      An empty dict means the same as None.
    group_labels: list of dict, VM group label matchers, or None.
    zones: list of zones, VM zone matchers, or None.
    instances: list of instances, instance name matchers, or None.

  Returns:
    Updated ops agents policy.
  """
    updated_description = ops_agents_policy.description if description is None else description
    updated_etag = ops_agents_policy.etag if etag is None else etag
    assignment = ops_agents_policy.assignment
    updated_assignment = OpsAgentPolicy.Assignment(group_labels=assignment.group_labels if group_labels is None else group_labels, zones=assignment.zones if zones is None else zones, instances=assignment.instances if instances is None else instances, os_types=CreateOsTypes(os_types) or assignment.os_types)
    updated_agent_rules = CreateAgentRules(agent_rules) or ops_agents_policy.agent_rules
    return OpsAgentPolicy(assignment=updated_assignment, agent_rules=updated_agent_rules, description=updated_description, etag=updated_etag, name=ops_agents_policy.id, update_time=None, create_time=ops_agents_policy.create_time)