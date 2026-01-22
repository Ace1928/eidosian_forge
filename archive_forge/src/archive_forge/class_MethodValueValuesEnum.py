from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MethodValueValuesEnum(_messages.Enum):
    """Optional. Specify if the threshold is used for probability or severity
    score. If not specified, the threshold is used for probability score.

    Values:
      HARM_BLOCK_METHOD_UNSPECIFIED: The harm block method is unspecified.
      SEVERITY: The harm block method uses both probability and severity
        scores.
      PROBABILITY: The harm block method uses the probability score.
    """
    HARM_BLOCK_METHOD_UNSPECIFIED = 0
    SEVERITY = 1
    PROBABILITY = 2