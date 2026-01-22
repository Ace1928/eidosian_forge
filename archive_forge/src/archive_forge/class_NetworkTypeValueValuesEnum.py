from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NetworkTypeValueValuesEnum(_messages.Enum):
    """Type of network.

    Values:
      TYPE_UNSPECIFIED: Unspecified value.
      CLIENT: Client network, a network peered to a Google Cloud VPC.
      PRIVATE: Private network, a network local to the Bare Metal Solution
        environment.
    """
    TYPE_UNSPECIFIED = 0
    CLIENT = 1
    PRIVATE = 2