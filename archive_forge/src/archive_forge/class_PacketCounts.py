from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PacketCounts(_messages.Message):
    """Containing a collection of interface-related statistics objects.

  Fields:
    inboundDiscards: The number of inbound packets that were chosen to be
      discarded even though no errors had been detected to prevent their being
      deliverable.
    inboundErrors: The number of inbound packets that contained errors.
    inboundUnicast: The number of packets that are delivered.
    outboundDiscards: The number of outbound packets that were chosen to be
      discarded even though no errors had been detected to prevent their being
      transmitted.
    outboundErrors: The number of outbound packets that could not be
      transmitted because of errors.
    outboundUnicast: The total number of packets that are requested be
      transmitted.
  """
    inboundDiscards = _messages.IntegerField(1)
    inboundErrors = _messages.IntegerField(2)
    inboundUnicast = _messages.IntegerField(3)
    outboundDiscards = _messages.IntegerField(4)
    outboundErrors = _messages.IntegerField(5)
    outboundUnicast = _messages.IntegerField(6)