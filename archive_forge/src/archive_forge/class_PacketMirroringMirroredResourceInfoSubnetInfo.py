from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PacketMirroringMirroredResourceInfoSubnetInfo(_messages.Message):
    """A PacketMirroringMirroredResourceInfoSubnetInfo object.

  Fields:
    canonicalUrl: [Output Only] Unique identifier for the subnetwork; defined
      by the server.
    url: Resource URL to the subnetwork for which traffic from/to all VM
      instances will be mirrored.
  """
    canonicalUrl = _messages.StringField(1)
    url = _messages.StringField(2)