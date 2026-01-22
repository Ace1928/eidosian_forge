import dataclasses
import enum
import json
import sys
from typing import Any, Mapping, Optional
from apitools.base.py import encoding
from googlecloudsdk.generated_clients.apis.osconfig.v1 import osconfig_v1_messages
@dataclasses.dataclass(repr=False)
class OpsAgentsPolicy(object):
    """An Ops Agent policy encapsulates the underlying VMM Policy.

  Attr:
    agents_rule: the agents rule to be applied to VMs.
    instance_filter:
      [InstanceFilter](https://cloud.google.com/compute/docs/osconfig/rest/v1/projects.locations.osPolicyAssignments#InstanceFilter)
      Filters to select target VMs for an assignment. Only Ops Agent supported
      [osShortName](https://cloud.google.com/compute/docs/osconfig/rest/v1/projects.locations.osPolicyAssignments#inventory)
      values are allowed.
  """

    @dataclasses.dataclass(repr=False)
    class AgentsRule(object):
        """An Ops agents rule contains package state, and version.

    Attr:
      version: agent version, e.g. 'latest', '2.52.1'.
      package_state: desired state for the package.
    """

        class PackageState(*_StrEnum):
            INSTALLED = 'installed'
            REMOVED = 'removed'
        version: Optional[str]
        package_state: PackageState = PackageState.INSTALLED

        def __repr__(self) -> str:
            """JSON single line format string."""
            return self.ToJson()

        def ToJson(self) -> str:
            """JSON single line format string."""
            return json.dumps(self.__dict__, separators=(',', ':'), default=str, sort_keys=True)
    agents_rule: AgentsRule
    instance_filter: osconfig_v1_messages.OSPolicyAssignmentInstanceFilter

    def __repr__(self) -> str:
        """JSON single line format string representation for testing."""
        policy_map = {'agents_rule': self.agents_rule, 'instance_filter': encoding.MessageToPyValue(self.instance_filter)}
        return json.dumps(policy_map, default=lambda o: o.__dict__, separators=(',', ':'), sort_keys=True)