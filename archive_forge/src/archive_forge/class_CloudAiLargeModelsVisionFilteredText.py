from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudAiLargeModelsVisionFilteredText(_messages.Message):
    """Details for filtered input text.

  Enums:
    CategoryValueValuesEnum: Confidence level
    ConfidenceValueValuesEnum: Filtered category

  Fields:
    category: Confidence level
    confidence: Filtered category
    prompt: Input prompt
    score: Score for category
  """

    class CategoryValueValuesEnum(_messages.Enum):
        """Confidence level

    Values:
      RAI_CATEGORY_UNSPECIFIED: <no description>
      OBSCENE: <no description>
      SEXUALLY_EXPLICIT: Porn
      IDENTITY_ATTACK: Hate
      VIOLENCE_ABUSE: <no description>
      CSAI: <no description>
      SPII: <no description>
      CELEBRITY: <no description>
      FACE_IMG: <no description>
      WATERMARK_IMG: <no description>
      MEMORIZATION_IMG: <no description>
      CSAI_IMG: <no description>
      PORN_IMG: <no description>
      VIOLENCE_IMG: <no description>
      CHILD_IMG: <no description>
      TOXIC: <no description>
      SENSITIVE_WORD: <no description>
      PERSON_IMG: <no description>
      ICA_IMG: <no description>
      SEXUAL_IMG: <no description>
      IU_IMG: <no description>
      RACY_IMG: <no description>
      PEDO_IMG: <no description>
      DEATH_HARM_TRAGEDY: SafetyAttributes returned but not filtered on
      HEALTH: <no description>
      FIREARMS_WEAPONS: <no description>
      RELIGIOUS_BELIEF: <no description>
      ILLICIT_DRUGS: <no description>
      WAR_CONFLICT: <no description>
      POLITICS: <no description>
      HATE_SYMBOL_IMG: End of list
      CHILD_TEXT: <no description>
      DANGEROUS_CONTENT: Text category from SafetyCat v3
      RECITATION_TEXT: <no description>
      CELEBRITY_IMG: <no description>
    """
        RAI_CATEGORY_UNSPECIFIED = 0
        OBSCENE = 1
        SEXUALLY_EXPLICIT = 2
        IDENTITY_ATTACK = 3
        VIOLENCE_ABUSE = 4
        CSAI = 5
        SPII = 6
        CELEBRITY = 7
        FACE_IMG = 8
        WATERMARK_IMG = 9
        MEMORIZATION_IMG = 10
        CSAI_IMG = 11
        PORN_IMG = 12
        VIOLENCE_IMG = 13
        CHILD_IMG = 14
        TOXIC = 15
        SENSITIVE_WORD = 16
        PERSON_IMG = 17
        ICA_IMG = 18
        SEXUAL_IMG = 19
        IU_IMG = 20
        RACY_IMG = 21
        PEDO_IMG = 22
        DEATH_HARM_TRAGEDY = 23
        HEALTH = 24
        FIREARMS_WEAPONS = 25
        RELIGIOUS_BELIEF = 26
        ILLICIT_DRUGS = 27
        WAR_CONFLICT = 28
        POLITICS = 29
        HATE_SYMBOL_IMG = 30
        CHILD_TEXT = 31
        DANGEROUS_CONTENT = 32
        RECITATION_TEXT = 33
        CELEBRITY_IMG = 34

    class ConfidenceValueValuesEnum(_messages.Enum):
        """Filtered category

    Values:
      CONFIDENCE_UNSPECIFIED: <no description>
      CONFIDENCE_LOW: <no description>
      CONFIDENCE_MEDIUM: <no description>
      CONFIDENCE_HIGH: <no description>
    """
        CONFIDENCE_UNSPECIFIED = 0
        CONFIDENCE_LOW = 1
        CONFIDENCE_MEDIUM = 2
        CONFIDENCE_HIGH = 3
    category = _messages.EnumField('CategoryValueValuesEnum', 1)
    confidence = _messages.EnumField('ConfidenceValueValuesEnum', 2)
    prompt = _messages.StringField(3)
    score = _messages.FloatField(4)