from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class StrategyValueValuesEnum(_messages.Enum):
    """Optional. This determines which type of scheduling strategy to use.

    Values:
      STRATEGY_UNSPECIFIED: Strategy will default to ON_DEMAND.
      ON_DEMAND: Regular on-demand provisioning strategy.
      LOW_COST: Low cost by making potential use of spot resources.
    """
    STRATEGY_UNSPECIFIED = 0
    ON_DEMAND = 1
    LOW_COST = 2