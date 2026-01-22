from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import json
import re
from googlecloudsdk.api_lib.compute.instances.ops_agents import exceptions
from googlecloudsdk.api_lib.compute.instances.ops_agents import ops_agents_policy as agent_policy
from googlecloudsdk.core import log
class AgentVersionAndEnableAutoupgradeConflictError(exceptions.PolicyValidationError):
    """Raised when agent version is pinned but autoupgrade is enabled."""

    def __init__(self, version):
        super(AgentVersionAndEnableAutoupgradeConflictError, self).__init__('An agent can not be pinned to the specific version [{}] when [enable-autoupgrade] is set to true for that agent.'.format(version))