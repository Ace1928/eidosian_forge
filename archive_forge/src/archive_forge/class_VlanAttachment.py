from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class VlanAttachment(_messages.Message):
    """VLAN attachment details.

  Fields:
    id: Immutable. The identifier of the attachment within vrf.
    interconnectAttachment: Optional. The name of the vlan attachment within
      vrf. This is of the form projects/{project_number}/regions/{region}/inte
      rconnectAttachments/{interconnect_attachment}
    pairingKey: Input only. Pairing key.
    peerIp: The peer IP of the attachment.
    peerVlanId: The peer vlan ID of the attachment.
    qosPolicy: The QOS policy applied to this VLAN attachment. This value
      should be preferred to using qos at vrf level.
    routerIp: The router IP of the attachment.
  """
    id = _messages.StringField(1)
    interconnectAttachment = _messages.StringField(2)
    pairingKey = _messages.StringField(3)
    peerIp = _messages.StringField(4)
    peerVlanId = _messages.IntegerField(5)
    qosPolicy = _messages.MessageField('QosPolicy', 6)
    routerIp = _messages.StringField(7)