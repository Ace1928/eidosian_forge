from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MultiTrialAlgorithmValueValuesEnum(_messages.Enum):
    """The multi-trial Neural Architecture Search (NAS) algorithm type.
    Defaults to `REINFORCEMENT_LEARNING`.

    Values:
      MULTI_TRIAL_ALGORITHM_UNSPECIFIED: Defaults to `REINFORCEMENT_LEARNING`.
      REINFORCEMENT_LEARNING: The Reinforcement Learning Algorithm for Multi-
        trial Neural Architecture Search (NAS).
      GRID_SEARCH: The Grid Search Algorithm for Multi-trial Neural
        Architecture Search (NAS).
    """
    MULTI_TRIAL_ALGORITHM_UNSPECIFIED = 0
    REINFORCEMENT_LEARNING = 1
    GRID_SEARCH = 2