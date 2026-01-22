from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class FeedbackTypeValueValuesEnum(_messages.Enum):
    """Required. The type of feedback being submitted.

    Values:
      FEEDBACK_TYPE_UNSPECIFIED: Unspecified feedback type.
      DETECTION_FALSE_POSITIVE: Feedback identifying an incorrect
        classification by an ML model.
      DETECTION_FALSE_NEGATIVE: Feedback identifying a classification by an ML
        model that was missed.
    """
    FEEDBACK_TYPE_UNSPECIFIED = 0
    DETECTION_FALSE_POSITIVE = 1
    DETECTION_FALSE_NEGATIVE = 2