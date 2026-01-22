from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowCxV3TestRunDifference(_messages.Message):
    """The description of differences between original and replayed agent
  output.

  Enums:
    TypeValueValuesEnum: The type of diff.

  Fields:
    description: A human readable description of the diff, showing the actual
      output vs expected output.
    type: The type of diff.
  """

    class TypeValueValuesEnum(_messages.Enum):
        """The type of diff.

    Values:
      DIFF_TYPE_UNSPECIFIED: Should never be used.
      INTENT: The intent.
      PAGE: The page.
      PARAMETERS: The parameters.
      UTTERANCE: The message utterance.
      FLOW: The flow.
    """
        DIFF_TYPE_UNSPECIFIED = 0
        INTENT = 1
        PAGE = 2
        PARAMETERS = 3
        UTTERANCE = 4
        FLOW = 5
    description = _messages.StringField(1)
    type = _messages.EnumField('TypeValueValuesEnum', 2)