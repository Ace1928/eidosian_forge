from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class VerticalAlignmentValueValuesEnum(_messages.Enum):
    """The vertical alignment of both the title and content

    Values:
      VERTICAL_ALIGNMENT_UNSPECIFIED: No vertical alignment specified, will
        default to V_TOP
      V_TOP: Top-align
      V_CENTER: Center-align
      V_BOTTOM: Bottom-align
    """
    VERTICAL_ALIGNMENT_UNSPECIFIED = 0
    V_TOP = 1
    V_CENTER = 2
    V_BOTTOM = 3