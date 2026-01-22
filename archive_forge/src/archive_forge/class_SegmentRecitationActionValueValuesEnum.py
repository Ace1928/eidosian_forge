from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SegmentRecitationActionValueValuesEnum(_messages.Enum):
    """SegmentRecitationActionValueValuesEnum enum type.

    Values:
      ACTION_UNSPECIFIED: <no description>
      CITE: indicate that attribution must be shown for a Segment
      BLOCK: indicate that a Segment should be blocked from being used
      NO_ACTION: for tagging high-frequency code snippets
      EXEMPT_FOUND_IN_PROMPT: The recitation was found in prompt and is
        exempted from overall results
    """
    ACTION_UNSPECIFIED = 0
    CITE = 1
    BLOCK = 2
    NO_ACTION = 3
    EXEMPT_FOUND_IN_PROMPT = 4