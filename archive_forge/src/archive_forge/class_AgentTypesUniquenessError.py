from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import json
import re
from googlecloudsdk.api_lib.compute.instances.ops_agents import exceptions
from googlecloudsdk.api_lib.compute.instances.ops_agents import ops_agents_policy as agent_policy
from googlecloudsdk.core import log
class AgentTypesUniquenessError(exceptions.PolicyValidationError):
    """Raised when agent type is not unique."""

    def __init__(self, agent_type):
        super(AgentTypesUniquenessError, self).__init__('At most one agent with type [{}] is allowed.'.format(agent_type))