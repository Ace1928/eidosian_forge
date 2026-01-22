from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NetworkConfigValueValuesEnum(_messages.Enum):
    """The type of network configuration on the instance.

    Values:
      NETWORKCONFIG_UNSPECIFIED: The unspecified network configuration.
      SINGLE_VLAN: Instance part of single client network and single private
        network.
      MULTI_VLAN: Instance part of multiple (or single) client networks and
        private networks.
    """
    NETWORKCONFIG_UNSPECIFIED = 0
    SINGLE_VLAN = 1
    MULTI_VLAN = 2