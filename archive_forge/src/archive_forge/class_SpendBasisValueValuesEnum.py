from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SpendBasisValueValuesEnum(_messages.Enum):
    """Optional. The type of basis used to determine if spend has passed the
    threshold. Behavior defaults to CURRENT_SPEND if not set.

    Values:
      BASIS_UNSPECIFIED: Unspecified threshold basis.
      CURRENT_SPEND: Use current spend as the basis for comparison against the
        threshold.
      FORECASTED_SPEND: Use forecasted spend for the period as the basis for
        comparison against the threshold. FORECASTED_SPEND can only be set
        when the budget's time period is a Filter.calendar_period. It cannot
        be set in combination with Filter.custom_period.
    """
    BASIS_UNSPECIFIED = 0
    CURRENT_SPEND = 1
    FORECASTED_SPEND = 2