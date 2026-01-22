from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PreservedStatePreservedNetworkIpIpAddress(_messages.Message):
    """A PreservedStatePreservedNetworkIpIpAddress object.

  Fields:
    address: The URL of the reservation for this IP address.
    literal: An IPv4 internal network address to assign to the instance for
      this network interface.
  """
    address = _messages.StringField(1)
    literal = _messages.StringField(2)