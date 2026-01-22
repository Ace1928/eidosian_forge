from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class FadeTypeValueValuesEnum(_messages.Enum):
    """Required. Type of fade animation: `FADE_IN` or `FADE_OUT`.

    Values:
      FADE_TYPE_UNSPECIFIED: The fade type is not specified.
      FADE_IN: Fade the overlay object into view.
      FADE_OUT: Fade the overlay object out of view.
    """
    FADE_TYPE_UNSPECIFIED = 0
    FADE_IN = 1
    FADE_OUT = 2