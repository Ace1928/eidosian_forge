from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class InterconnectDiagnosticsLinkStatus(_messages.Message):
    """A InterconnectDiagnosticsLinkStatus object.

  Enums:
    OperationalStatusValueValuesEnum: The operational status of the link.

  Fields:
    arpCaches: A list of InterconnectDiagnostics.ARPEntry objects, describing
      the ARP neighbor entries seen on this link. This will be empty if the
      link is bundled
    circuitId: The unique ID for this link assigned during turn up by Google.
    googleDemarc: The Demarc address assigned by Google and provided in the
      LoA.
    lacpStatus: A InterconnectDiagnosticsLinkLACPStatus attribute.
    macsec: Describes the status of MACsec encryption on this link.
    operationalStatus: The operational status of the link.
    receivingOpticalPower: An InterconnectDiagnostics.LinkOpticalPower object,
      describing the current value and status of the received light level.
    transmittingOpticalPower: An InterconnectDiagnostics.LinkOpticalPower
      object, describing the current value and status of the transmitted light
      level.
  """

    class OperationalStatusValueValuesEnum(_messages.Enum):
        """The operational status of the link.

    Values:
      LINK_OPERATIONAL_STATUS_DOWN: The interface is unable to communicate
        with the remote end.
      LINK_OPERATIONAL_STATUS_UP: The interface has low level communication
        with the remote end.
    """
        LINK_OPERATIONAL_STATUS_DOWN = 0
        LINK_OPERATIONAL_STATUS_UP = 1
    arpCaches = _messages.MessageField('InterconnectDiagnosticsARPEntry', 1, repeated=True)
    circuitId = _messages.StringField(2)
    googleDemarc = _messages.StringField(3)
    lacpStatus = _messages.MessageField('InterconnectDiagnosticsLinkLACPStatus', 4)
    macsec = _messages.MessageField('InterconnectDiagnosticsMacsecStatus', 5)
    operationalStatus = _messages.EnumField('OperationalStatusValueValuesEnum', 6)
    receivingOpticalPower = _messages.MessageField('InterconnectDiagnosticsLinkOpticalPower', 7)
    transmittingOpticalPower = _messages.MessageField('InterconnectDiagnosticsLinkOpticalPower', 8)