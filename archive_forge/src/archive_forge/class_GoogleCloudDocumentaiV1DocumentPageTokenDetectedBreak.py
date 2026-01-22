from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDocumentaiV1DocumentPageTokenDetectedBreak(_messages.Message):
    """Detected break at the end of a Token.

  Enums:
    TypeValueValuesEnum: Detected break type.

  Fields:
    type: Detected break type.
  """

    class TypeValueValuesEnum(_messages.Enum):
        """Detected break type.

    Values:
      TYPE_UNSPECIFIED: Unspecified break type.
      SPACE: A single whitespace.
      WIDE_SPACE: A wider whitespace.
      HYPHEN: A hyphen that indicates that a token has been split across
        lines.
    """
        TYPE_UNSPECIFIED = 0
        SPACE = 1
        WIDE_SPACE = 2
        HYPHEN = 3
    type = _messages.EnumField('TypeValueValuesEnum', 1)