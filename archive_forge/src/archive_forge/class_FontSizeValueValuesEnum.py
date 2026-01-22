from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class FontSizeValueValuesEnum(_messages.Enum):
    """Font sizes for both the title and content. The title will still be
    larger relative to the content.

    Values:
      FONT_SIZE_UNSPECIFIED: No font size specified, will default to FS_LARGE
      FS_EXTRA_SMALL: Extra small font size
      FS_SMALL: Small font size
      FS_MEDIUM: Medium font size
      FS_LARGE: Large font size
      FS_EXTRA_LARGE: Extra large font size
    """
    FONT_SIZE_UNSPECIFIED = 0
    FS_EXTRA_SMALL = 1
    FS_SMALL = 2
    FS_MEDIUM = 3
    FS_LARGE = 4
    FS_EXTRA_LARGE = 5