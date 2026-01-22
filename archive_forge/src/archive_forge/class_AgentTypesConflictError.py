from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import json
import re
from googlecloudsdk.api_lib.compute.instances.ops_agents import exceptions
from googlecloudsdk.api_lib.compute.instances.ops_agents import ops_agents_policy as agent_policy
from googlecloudsdk.core import log
class AgentTypesConflictError(exceptions.PolicyValidationError):
    """Raised when agent type is ops-agent and another agent type is specified."""

    def __init__(self):
        super(AgentTypesConflictError, self).__init__('An agent with type [ops-agent] is detected. No other agent type is allowed. The Ops Agent has both a logging module and a metrics module already.')