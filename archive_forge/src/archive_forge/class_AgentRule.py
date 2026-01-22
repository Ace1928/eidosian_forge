from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import enum
import json
import sys
from googlecloudsdk.core.resource import resource_property
class AgentRule(object):
    """An Ops agent rule contains agent type, version, enable_autoupgrade."""

    class Type(*_StrEnum):
        LOGGING = 'logging'
        METRICS = 'metrics'
        OPS_AGENT = 'ops-agent'

    class PackageState(*_StrEnum):
        INSTALLED = 'installed'
        REMOVED = 'removed'

    class Version(*_StrEnum):
        LATEST_OF_ALL = 'latest'
        CURRENT_MAJOR = 'current-major'

    def __init__(self, agent_type, enable_autoupgrade, version=Version.CURRENT_MAJOR, package_state=PackageState.INSTALLED):
        """Initialize AgentRule instance.

      Args:
        agent_type: Type, agent type to be installed.
        enable_autoupgrade: bool, enable autoupgrade for the package or
          not.
        version: str, agent version, e.g. 'latest', '5.5.2', '5.*.*'.
        package_state: Optional PackageState, desiredState for the package.
      """
        self.type = agent_type
        self.enable_autoupgrade = enable_autoupgrade
        self.version = version
        self.package_state = package_state

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def ToJson(self):
        """Generate JSON with camel-cased key."""
        key_camel_cased_dict = {resource_property.ConvertToCamelCase(key): value for key, value in self.__dict__.items()}
        return json.dumps(key_camel_cased_dict, default=str, sort_keys=True)