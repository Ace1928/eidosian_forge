from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SafetyRating(_messages.Message):
    """Safety rating for a piece of content. The safety rating contains the
  category of harm and the harm probability level in that category for a piece
  of content. Content is classified for safety across a number of harm
  categories and the probability of the harm classification is included here.

  Enums:
    CategoryValueValuesEnum: Required. The category for this rating.
    ProbabilityValueValuesEnum: Required. The probability of harm for this
      content.

  Fields:
    blocked: Was this content blocked because of this rating?
    category: Required. The category for this rating.
    probability: Required. The probability of harm for this content.
  """

    class CategoryValueValuesEnum(_messages.Enum):
        """Required. The category for this rating.

    Values:
      HARM_CATEGORY_UNSPECIFIED: The harm category is unspecified.
      HARM_CATEGORY_HATE_SPEECH: The harm category is hate speech.
      HARM_CATEGORY_DANGEROUS_CONTENT: The harm category is dangerous content.
      HARM_CATEGORY_HARASSMENT: The harm category is harassment.
      HARM_CATEGORY_SEXUALLY_EXPLICIT: The harm category is sexually explicit
        content.
    """
        HARM_CATEGORY_UNSPECIFIED = 0
        HARM_CATEGORY_HATE_SPEECH = 1
        HARM_CATEGORY_DANGEROUS_CONTENT = 2
        HARM_CATEGORY_HARASSMENT = 3
        HARM_CATEGORY_SEXUALLY_EXPLICIT = 4

    class ProbabilityValueValuesEnum(_messages.Enum):
        """Required. The probability of harm for this content.

    Values:
      HARM_PROBABILITY_UNSPECIFIED: Probability is unspecified.
      NEGLIGIBLE: Content has a negligible chance of being unsafe.
      LOW: Content has a low chance of being unsafe.
      MEDIUM: Content has a medium chance of being unsafe.
      HIGH: Content has a high chance of being unsafe.
    """
        HARM_PROBABILITY_UNSPECIFIED = 0
        NEGLIGIBLE = 1
        LOW = 2
        MEDIUM = 3
        HIGH = 4
    blocked = _messages.BooleanField(1)
    category = _messages.EnumField('CategoryValueValuesEnum', 2)
    probability = _messages.EnumField('ProbabilityValueValuesEnum', 3)