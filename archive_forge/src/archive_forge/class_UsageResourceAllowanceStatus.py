from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class UsageResourceAllowanceStatus(_messages.Message):
    """Status of a usage ResourceAllowance.

  Enums:
    StateValueValuesEnum: Output only. ResourceAllowance state.

  Fields:
    limitStatus: Output only. ResourceAllowance consumption status for usage
      resources.
    report: Output only. The report of ResourceAllowance consumptions in a
      time period.
    state: Output only. ResourceAllowance state.
  """

    class StateValueValuesEnum(_messages.Enum):
        """Output only. ResourceAllowance state.

    Values:
      RESOURCE_ALLOWANCE_STATE_UNSPECIFIED: Unspecified.
      RESOURCE_ALLOWANCE_ACTIVE: ResourceAllowance is active and in use.
      RESOURCE_ALLOWANCE_DEPLETED: ResourceAllowance limit is reached.
    """
        RESOURCE_ALLOWANCE_STATE_UNSPECIFIED = 0
        RESOURCE_ALLOWANCE_ACTIVE = 1
        RESOURCE_ALLOWANCE_DEPLETED = 2
    limitStatus = _messages.MessageField('LimitStatus', 1)
    report = _messages.MessageField('ConsumptionReport', 2)
    state = _messages.EnumField('StateValueValuesEnum', 3)