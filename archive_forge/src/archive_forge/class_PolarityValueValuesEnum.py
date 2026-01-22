from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PolarityValueValuesEnum(_messages.Enum):
    """Whether to only highlight pixels with positive contributions, negative
    or both. Defaults to POSITIVE.

    Values:
      POLARITY_UNSPECIFIED: Default value. This is the same as POSITIVE.
      POSITIVE: Highlights the pixels/outlines that were most influential to
        the model's prediction.
      NEGATIVE: Setting polarity to negative highlights areas that does not
        lead to the models's current prediction.
      BOTH: Shows both positive and negative attributions.
    """
    POLARITY_UNSPECIFIED = 0
    POSITIVE = 1
    NEGATIVE = 2
    BOTH = 3