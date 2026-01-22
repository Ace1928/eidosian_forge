from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import json
import re
from googlecloudsdk.api_lib.compute.instances.ops_agents import exceptions
from googlecloudsdk.api_lib.compute.instances.ops_agents import ops_agents_policy as agent_policy
from googlecloudsdk.core import log
def _ValidateAgentRule(agent_rule):
    """Validates semantics of an individual OpsAgentPolicy.AgentRule.

  This validation happens after the arg parsing stage. At this point, we can
  assume that the field is an OpsAgentPolicy.AgentRule object.

  Args:
    agent_rule: OpsAgentPolicy.AgentRule. The agent rule to enforce by the Ops
      Agents policy.

  Returns:
    An empty list if the validation passes. A list of errors from the following
    list if the validation fails.
    * AgentVersionInvalidFormatError:
      Agent version format is invalid.
    * AgentVersionAndEnableAutoupgradeConflictError:
      Agent version is pinned but autoupgrade is enabled.
  """
    return _ValidateAgentVersion(agent_rule.type, agent_rule.version) + _ValidateAgentVersionAndEnableAutoupgrade(agent_rule.version, agent_rule.enable_autoupgrade)