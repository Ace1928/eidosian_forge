from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class LinkLayerAddress(_messages.Message):
    """LinkLayerAddress contains an IP address and corresponding link-layer
  address.

  Fields:
    ipAddress: The IP address of this neighbor.
    macAddress: The MAC address of this neighbor.
  """
    ipAddress = _messages.StringField(1)
    macAddress = _messages.StringField(2)