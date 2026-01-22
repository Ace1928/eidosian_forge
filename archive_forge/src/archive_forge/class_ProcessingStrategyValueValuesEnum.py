from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ProcessingStrategyValueValuesEnum(_messages.Enum):
    """Processing strategy to use for this request.

    Values:
      PROCESSING_STRATEGY_UNSPECIFIED: Default value for the processing
        strategy. The request is processed as soon as its received.
      DYNAMIC_BATCHING: If selected, processes the request during lower
        utilization periods for a price discount. The request is fulfilled
        within 24 hours.
    """
    PROCESSING_STRATEGY_UNSPECIFIED = 0
    DYNAMIC_BATCHING = 1