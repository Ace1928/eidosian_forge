from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class OptimizationValueValuesEnum(_messages.Enum):
    """Optional. The optimization strategy of the job. The default is
    `AUTODETECT`.

    Values:
      OPTIMIZATION_STRATEGY_UNSPECIFIED: The optimization strategy is not
        specified.
      AUTODETECT: Prioritize job processing speed.
      DISABLED: Disable all optimizations.
    """
    OPTIMIZATION_STRATEGY_UNSPECIFIED = 0
    AUTODETECT = 1
    DISABLED = 2