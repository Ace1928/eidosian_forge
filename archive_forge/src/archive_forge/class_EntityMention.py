from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class EntityMention(_messages.Message):
    """Represents a mention for an entity in the text. Currently, proper noun
  mentions are supported.

  Enums:
    TypeValueValuesEnum: The type of the entity mention.

  Fields:
    sentiment: For calls to AnalyzeEntitySentiment or if
      AnnotateTextRequest.Features.extract_entity_sentiment is set to true,
      this field will contain the sentiment expressed for this mention of the
      entity in the provided document.
    text: The mention text.
    type: The type of the entity mention.
  """

    class TypeValueValuesEnum(_messages.Enum):
        """The type of the entity mention.

    Values:
      TYPE_UNKNOWN: Unknown
      PROPER: Proper name
      COMMON: Common noun (or noun compound)
    """
        TYPE_UNKNOWN = 0
        PROPER = 1
        COMMON = 2
    sentiment = _messages.MessageField('Sentiment', 1)
    text = _messages.MessageField('TextSpan', 2)
    type = _messages.EnumField('TypeValueValuesEnum', 3)