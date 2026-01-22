from __future__ import absolute_import
from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DeterminismLevelValueValuesEnum(_messages.Enum):
    """Optional. The determinism level of the JavaScript UDF, if defined.

    Values:
      DETERMINISM_LEVEL_UNSPECIFIED: The determinism of the UDF is
        unspecified.
      DETERMINISTIC: The UDF is deterministic, meaning that 2 function calls
        with the same inputs always produce the same result, even across 2
        query runs.
      NOT_DETERMINISTIC: The UDF is not deterministic.
    """
    DETERMINISM_LEVEL_UNSPECIFIED = 0
    DETERMINISTIC = 1
    NOT_DETERMINISTIC = 2