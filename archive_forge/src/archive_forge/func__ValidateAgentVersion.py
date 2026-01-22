from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import json
import re
from googlecloudsdk.api_lib.compute.instances.ops_agents import exceptions
from googlecloudsdk.api_lib.compute.instances.ops_agents import ops_agents_policy as agent_policy
from googlecloudsdk.core import log
def _ValidateAgentVersion(agent_type, version):
    """Validates agent version format.

  This validation happens after the arg parsing stage. At this point, we can
  assume that the field is a valid string.

  Args:
    agent_type: str. The type of agent to be installed. Allowed values:
      * "logging"
      * "metrics"
    version: str. The version of agent. Allowed values:
      * "latest"
      * "current-major"
      * "[MAJOR_VERSION].*.*"
      * "[MAJOR_VERSION].[MINOR_VERSION].[PATCH_VERSION]"

  Returns:
    An empty list if the validation passes. A singleton list with one of
    the following errors if the validation fails.
    * AgentVersionInvalidFormatError:
      Agent version format is invalid.
    * AgentMajorVersionNotSupportedError:
      Agent's major version is not supported for the given agent type.
  """
    version_enum = agent_policy.OpsAgentPolicy.AgentRule.Version
    if version in {version_enum.LATEST_OF_ALL, version_enum.CURRENT_MAJOR}:
        return []
    valid_pin_res = {_PINNED_MAJOR_VERSION_RE, _PINNED_LEGACY_VERSION_RE, _PINNED_VERSION_RE}
    if not any((regex.search(version) for regex in valid_pin_res)):
        return [AgentVersionInvalidFormatError(version)]
    major_version = version.split('.')[0]
    if major_version not in _SUPPORTED_AGENT_MAJOR_VERSIONS[agent_type]:
        return [AgentUnsupportedMajorVersionError(agent_type, version)]
    return []