from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class HorizontalAlignmentValueValuesEnum(_messages.Enum):
    """The horizontal alignment of both the title and content

    Values:
      HORIZONTAL_ALIGNMENT_UNSPECIFIED: No horizontal alignment specified,
        will default to H_LEFT
      H_LEFT: Left-align
      H_CENTER: Center-align
      H_RIGHT: Right-align
    """
    HORIZONTAL_ALIGNMENT_UNSPECIFIED = 0
    H_LEFT = 1
    H_CENTER = 2
    H_RIGHT = 3