from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import json
import re
from googlecloudsdk.api_lib.compute.instances.ops_agents import exceptions
from googlecloudsdk.api_lib.compute.instances.ops_agents import ops_agents_policy as agent_policy
from googlecloudsdk.core import log
def _ValidateAgentRules(agent_rules):
    """Validates semantics of the ops-agents-policy.agent-rules field.

  This validation happens after the arg parsing stage. At this point, we can
  assume that the field is a list of OpsAgentPolicy.AgentRule object.

  Args:
    agent_rules: list of OpsAgentPolicy.AgentRule. The list of agent rules to be
      managed by the Ops Agents policy.

  Returns:
    An empty list if the validation passes. A list of errors from the following
    list if the validation fails.
    * AgentTypesUniquenessError:
      Multiple agents with the same type are specified.
    * AgentTypesConflictError:
      More than one agent type is specified when there is already a type
      ops-agent.
    * AgentVersionInvalidFormatError:
      Agent version format is invalid.
    * AgentVersionAndEnableAutoupgradeConflictError:
      Agent version is pinned but autoupgrade is enabled.
  """
    errors = _ValidateAgentTypesUniqueness(agent_rules)
    errors.extend(_ValidateAgentTypesConflict(agent_rules))
    for agent_rule in agent_rules:
        errors.extend(_ValidateAgentRule(agent_rule))
    return errors