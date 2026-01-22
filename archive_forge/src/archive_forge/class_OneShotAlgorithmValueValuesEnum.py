from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class OneShotAlgorithmValueValuesEnum(_messages.Enum):
    """Optional. The one-shot Neural Architecture Search (NAS) algorithm
    type. Defaults to `ONE_SHOT_ALGORITHM_REINFORCEMENT_LEARNING`.

    Values:
      ONE_SHOT_ALGORITHM_UNSPECIFIED: <no description>
      REINFORCEMENT_LEARNING: The Reinforcement Learning Algorithm for one-
        shot Neural Architecture Search (NAS).
    """
    ONE_SHOT_ALGORITHM_UNSPECIFIED = 0
    REINFORCEMENT_LEARNING = 1