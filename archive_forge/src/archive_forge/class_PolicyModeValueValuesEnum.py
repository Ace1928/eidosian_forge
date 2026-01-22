from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PolicyModeValueValuesEnum(_messages.Enum):
    """Optional. The policy mode of this hub. This field can be either PRESET
    or CUSTOM. If unspecified, the policy_mode defaults to PRESET.

    Values:
      POLICY_MODE_UNSPECIFIED: Policy mode is unspecified. It defaults to
        PRESET with preset_topology = MESH.
      PRESET: Hub uses one of the preset topologies.
      CUSTOM: Hub can freely specify the topology using groups and policy.
    """
    POLICY_MODE_UNSPECIFIED = 0
    PRESET = 1
    CUSTOM = 2