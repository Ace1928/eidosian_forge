from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudAiNlLlmProtoServiceRaiSignalInfluentialTerm(_messages.Message):
    """The influential term that could potentially block the response.

  Enums:
    SourceValueValuesEnum: The source of the influential term, prompt or
      response.

  Fields:
    beginOffset: The beginning offset of the influential term.
    confidence: The confidence score of the influential term.
    source: The source of the influential term, prompt or response.
    term: The influential term.
  """

    class SourceValueValuesEnum(_messages.Enum):
        """The source of the influential term, prompt or response.

    Values:
      SOURCE_UNSPECIFIED: Unspecified source.
      PROMPT: The influential term comes from the prompt.
      RESPONSE: The influential term comes from the response.
    """
        SOURCE_UNSPECIFIED = 0
        PROMPT = 1
        RESPONSE = 2
    beginOffset = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    confidence = _messages.FloatField(2, variant=_messages.Variant.FLOAT)
    source = _messages.EnumField('SourceValueValuesEnum', 3)
    term = _messages.StringField(4)