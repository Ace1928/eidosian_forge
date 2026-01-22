from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class InterconnectDiagnosticsARPEntry(_messages.Message):
    """Describing the ARP neighbor entries seen on this link

  Fields:
    ipAddress: The IP address of this ARP neighbor.
    macAddress: The MAC address of this ARP neighbor.
  """
    ipAddress = _messages.StringField(1)
    macAddress = _messages.StringField(2)