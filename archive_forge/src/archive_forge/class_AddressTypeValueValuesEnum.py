from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AddressTypeValueValuesEnum(_messages.Enum):
    """The type of address to reserve, either INTERNAL or EXTERNAL. If
    unspecified, defaults to EXTERNAL.

    Values:
      EXTERNAL: A publicly visible external IP address.
      INTERNAL: A private network IP address, for use with an Instance or
        Internal Load Balancer forwarding rule.
      UNSPECIFIED_TYPE: <no description>
    """
    EXTERNAL = 0
    INTERNAL = 1
    UNSPECIFIED_TYPE = 2