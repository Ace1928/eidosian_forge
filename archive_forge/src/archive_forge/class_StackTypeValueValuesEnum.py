from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class StackTypeValueValuesEnum(_messages.Enum):
    """The stack type for this network interface.

    Values:
      STACK_TYPE_UNSPECIFIED: Default should be STACK_TYPE_UNSPECIFIED.
      IPV4_ONLY: The network interface will be assigned IPv4 address.
      IPV4_IPV6: The network interface can have both IPv4 and IPv6 addresses.
    """
    STACK_TYPE_UNSPECIFIED = 0
    IPV4_ONLY = 1
    IPV4_IPV6 = 2