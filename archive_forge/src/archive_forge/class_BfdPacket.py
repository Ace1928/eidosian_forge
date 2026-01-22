from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BfdPacket(_messages.Message):
    """A BfdPacket object.

  Enums:
    DiagnosticValueValuesEnum: The diagnostic code specifies the local
      system's reason for the last change in session state. This allows remote
      systems to determine the reason that the previous session failed, for
      example. These diagnostic codes are specified in section 4.1 of RFC5880
    StateValueValuesEnum: The current BFD session state as seen by the
      transmitting system. These states are specified in section 4.1 of
      RFC5880

  Fields:
    authenticationPresent: The Authentication Present bit of the BFD packet.
      This is specified in section 4.1 of RFC5880
    controlPlaneIndependent: The Control Plane Independent bit of the BFD
      packet. This is specified in section 4.1 of RFC5880
    demand: The demand bit of the BFD packet. This is specified in section 4.1
      of RFC5880
    diagnostic: The diagnostic code specifies the local system's reason for
      the last change in session state. This allows remote systems to
      determine the reason that the previous session failed, for example.
      These diagnostic codes are specified in section 4.1 of RFC5880
    final: The Final bit of the BFD packet. This is specified in section 4.1
      of RFC5880
    length: The length of the BFD Control packet in bytes. This is specified
      in section 4.1 of RFC5880
    minEchoRxIntervalMs: The Required Min Echo RX Interval value in the BFD
      packet. This is specified in section 4.1 of RFC5880
    minRxIntervalMs: The Required Min RX Interval value in the BFD packet.
      This is specified in section 4.1 of RFC5880
    minTxIntervalMs: The Desired Min TX Interval value in the BFD packet. This
      is specified in section 4.1 of RFC5880
    multiplier: The detection time multiplier of the BFD packet. This is
      specified in section 4.1 of RFC5880
    multipoint: The multipoint bit of the BFD packet. This is specified in
      section 4.1 of RFC5880
    myDiscriminator: The My Discriminator value in the BFD packet. This is
      specified in section 4.1 of RFC5880
    poll: The Poll bit of the BFD packet. This is specified in section 4.1 of
      RFC5880
    state: The current BFD session state as seen by the transmitting system.
      These states are specified in section 4.1 of RFC5880
    version: The version number of the BFD protocol, as specified in section
      4.1 of RFC5880.
    yourDiscriminator: The Your Discriminator value in the BFD packet. This is
      specified in section 4.1 of RFC5880
  """

    class DiagnosticValueValuesEnum(_messages.Enum):
        """The diagnostic code specifies the local system's reason for the last
    change in session state. This allows remote systems to determine the
    reason that the previous session failed, for example. These diagnostic
    codes are specified in section 4.1 of RFC5880

    Values:
      ADMINISTRATIVELY_DOWN: <no description>
      CONCATENATED_PATH_DOWN: <no description>
      CONTROL_DETECTION_TIME_EXPIRED: <no description>
      DIAGNOSTIC_UNSPECIFIED: <no description>
      ECHO_FUNCTION_FAILED: <no description>
      FORWARDING_PLANE_RESET: <no description>
      NEIGHBOR_SIGNALED_SESSION_DOWN: <no description>
      NO_DIAGNOSTIC: <no description>
      PATH_DOWN: <no description>
      REVERSE_CONCATENATED_PATH_DOWN: <no description>
    """
        ADMINISTRATIVELY_DOWN = 0
        CONCATENATED_PATH_DOWN = 1
        CONTROL_DETECTION_TIME_EXPIRED = 2
        DIAGNOSTIC_UNSPECIFIED = 3
        ECHO_FUNCTION_FAILED = 4
        FORWARDING_PLANE_RESET = 5
        NEIGHBOR_SIGNALED_SESSION_DOWN = 6
        NO_DIAGNOSTIC = 7
        PATH_DOWN = 8
        REVERSE_CONCATENATED_PATH_DOWN = 9

    class StateValueValuesEnum(_messages.Enum):
        """The current BFD session state as seen by the transmitting system.
    These states are specified in section 4.1 of RFC5880

    Values:
      ADMIN_DOWN: <no description>
      DOWN: <no description>
      INIT: <no description>
      STATE_UNSPECIFIED: <no description>
      UP: <no description>
    """
        ADMIN_DOWN = 0
        DOWN = 1
        INIT = 2
        STATE_UNSPECIFIED = 3
        UP = 4
    authenticationPresent = _messages.BooleanField(1)
    controlPlaneIndependent = _messages.BooleanField(2)
    demand = _messages.BooleanField(3)
    diagnostic = _messages.EnumField('DiagnosticValueValuesEnum', 4)
    final = _messages.BooleanField(5)
    length = _messages.IntegerField(6, variant=_messages.Variant.UINT32)
    minEchoRxIntervalMs = _messages.IntegerField(7, variant=_messages.Variant.UINT32)
    minRxIntervalMs = _messages.IntegerField(8, variant=_messages.Variant.UINT32)
    minTxIntervalMs = _messages.IntegerField(9, variant=_messages.Variant.UINT32)
    multiplier = _messages.IntegerField(10, variant=_messages.Variant.UINT32)
    multipoint = _messages.BooleanField(11)
    myDiscriminator = _messages.IntegerField(12, variant=_messages.Variant.UINT32)
    poll = _messages.BooleanField(13)
    state = _messages.EnumField('StateValueValuesEnum', 14)
    version = _messages.IntegerField(15, variant=_messages.Variant.UINT32)
    yourDiscriminator = _messages.IntegerField(16, variant=_messages.Variant.UINT32)