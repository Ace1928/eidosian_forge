from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SampleStrategyValueValuesEnum(_messages.Enum):
    """Field to choose sampling strategy. Sampling strategy will decide which
    data should be selected for human labeling in every batch.

    Values:
      SAMPLE_STRATEGY_UNSPECIFIED: Default will be treated as UNCERTAINTY.
      UNCERTAINTY: Sample the most uncertain data to label.
    """
    SAMPLE_STRATEGY_UNSPECIFIED = 0
    UNCERTAINTY = 1