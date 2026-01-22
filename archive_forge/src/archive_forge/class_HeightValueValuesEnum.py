from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class HeightValueValuesEnum(_messages.Enum):
    """Required for cards with vertical orientation. The height of the media
    within a rich card with a vertical layout. For a standalone card with
    horizontal layout, height is not customizable, and this field is ignored.

    Values:
      HEIGHT_UNSPECIFIED: Not specified.
      SHORT: 112 DP.
      MEDIUM: 168 DP.
      TALL: 264 DP. Not available for rich card carousels when the card width
        is set to small.
    """
    HEIGHT_UNSPECIFIED = 0
    SHORT = 1
    MEDIUM = 2
    TALL = 3