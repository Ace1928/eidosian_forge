from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class InterconnectDiagnosticsLinkOpticalPower(_messages.Message):
    """A InterconnectDiagnosticsLinkOpticalPower object.

  Enums:
    StateValueValuesEnum: The status of the current value when compared to the
      warning and alarm levels for the receiving or transmitting transceiver.
      Possible states include: - OK: The value has not crossed a warning
      threshold. - LOW_WARNING: The value has crossed below the low warning
      threshold. - HIGH_WARNING: The value has crossed above the high warning
      threshold. - LOW_ALARM: The value has crossed below the low alarm
      threshold. - HIGH_ALARM: The value has crossed above the high alarm
      threshold.

  Fields:
    state: The status of the current value when compared to the warning and
      alarm levels for the receiving or transmitting transceiver. Possible
      states include: - OK: The value has not crossed a warning threshold. -
      LOW_WARNING: The value has crossed below the low warning threshold. -
      HIGH_WARNING: The value has crossed above the high warning threshold. -
      LOW_ALARM: The value has crossed below the low alarm threshold. -
      HIGH_ALARM: The value has crossed above the high alarm threshold.
    value: Value of the current receiving or transmitting optical power, read
      in dBm. Take a known good optical value, give it a 10% margin and
      trigger warnings relative to that value. In general, a -7dBm warning and
      a -11dBm alarm are good optical value estimates for most links.
  """

    class StateValueValuesEnum(_messages.Enum):
        """The status of the current value when compared to the warning and alarm
    levels for the receiving or transmitting transceiver. Possible states
    include: - OK: The value has not crossed a warning threshold. -
    LOW_WARNING: The value has crossed below the low warning threshold. -
    HIGH_WARNING: The value has crossed above the high warning threshold. -
    LOW_ALARM: The value has crossed below the low alarm threshold. -
    HIGH_ALARM: The value has crossed above the high alarm threshold.

    Values:
      HIGH_ALARM: The value has crossed above the high alarm threshold.
      HIGH_WARNING: The value of the current optical power has crossed above
        the high warning threshold.
      LOW_ALARM: The value of the current optical power has crossed below the
        low alarm threshold.
      LOW_WARNING: The value of the current optical power has crossed below
        the low warning threshold.
      OK: The value of the current optical power has not crossed a warning
        threshold.
    """
        HIGH_ALARM = 0
        HIGH_WARNING = 1
        LOW_ALARM = 2
        LOW_WARNING = 3
        OK = 4
    state = _messages.EnumField('StateValueValuesEnum', 1)
    value = _messages.FloatField(2, variant=_messages.Variant.FLOAT)