from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TagStateValueValuesEnum(_messages.Enum):
    """Match versions by tag status.

    Values:
      TAG_STATE_UNSPECIFIED: Tag status not specified.
      TAGGED: Applies to tagged versions only.
      UNTAGGED: Applies to untagged versions only.
      ANY: Applies to all versions.
    """
    TAG_STATE_UNSPECIFIED = 0
    TAGGED = 1
    UNTAGGED = 2
    ANY = 3