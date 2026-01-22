from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import json
import re
from googlecloudsdk.api_lib.compute.instances.ops_agents import exceptions
from googlecloudsdk.api_lib.compute.instances.ops_agents import ops_agents_policy as agent_policy
from googlecloudsdk.core import log
class AgentUnsupportedMajorVersionError(exceptions.PolicyValidationError):
    """Raised when agent's major version is not supported."""

    def __init__(self, agent_type, version):
        supported_versions = _SUPPORTED_AGENT_MAJOR_VERSIONS[agent_type]
        super(AgentUnsupportedMajorVersionError, self).__init__('The agent major version [{}] is not supported for agent type [{}]. Supported major versions are: {}'.format(version, agent_type, ', '.join(supported_versions)))