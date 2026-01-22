from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AllowPacketMirroringValueValuesEnum(_messages.Enum):
    """Specifies whether Packet Mirroring 1.0 is supported.

    Values:
      PACKET_MIRRORING_ALLOWED: <no description>
      PACKET_MIRRORING_BLOCKED: <no description>
      PACKET_MIRRORING_UNSPECIFIED: <no description>
    """
    PACKET_MIRRORING_ALLOWED = 0
    PACKET_MIRRORING_BLOCKED = 1
    PACKET_MIRRORING_UNSPECIFIED = 2