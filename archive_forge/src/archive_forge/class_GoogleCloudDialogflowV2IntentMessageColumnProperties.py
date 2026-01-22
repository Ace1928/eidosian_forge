from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowV2IntentMessageColumnProperties(_messages.Message):
    """Column properties for TableCard.

  Enums:
    HorizontalAlignmentValueValuesEnum: Optional. Defines text alignment for
      all cells in this column.

  Fields:
    header: Required. Column heading.
    horizontalAlignment: Optional. Defines text alignment for all cells in
      this column.
  """

    class HorizontalAlignmentValueValuesEnum(_messages.Enum):
        """Optional. Defines text alignment for all cells in this column.

    Values:
      HORIZONTAL_ALIGNMENT_UNSPECIFIED: Text is aligned to the leading edge of
        the column.
      LEADING: Text is aligned to the leading edge of the column.
      CENTER: Text is centered in the column.
      TRAILING: Text is aligned to the trailing edge of the column.
    """
        HORIZONTAL_ALIGNMENT_UNSPECIFIED = 0
        LEADING = 1
        CENTER = 2
        TRAILING = 3
    header = _messages.StringField(1)
    horizontalAlignment = _messages.EnumField('HorizontalAlignmentValueValuesEnum', 2)