from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudAiNlLlmProtoServiceRaiSignal(_messages.Message):
    """An RAI signal for a single category.

  Enums:
    ConfidenceValueValuesEnum: The confidence level for the RAI category.
    RaiCategoryValueValuesEnum: The RAI category.

  Fields:
    confidence: The confidence level for the RAI category.
    flagged: Whether the category is flagged as being present. Currently, this
      is set to true if score >= 0.5.
    influentialTerms: The influential terms that could potentially block the
      response.
    raiCategory: The RAI category.
    score: The score for the category, in the range [0.0, 1.0].
  """

    class ConfidenceValueValuesEnum(_messages.Enum):
        """The confidence level for the RAI category.

    Values:
      CONFIDENCE_UNSPECIFIED: <no description>
      CONFIDENCE_NONE: <no description>
      CONFIDENCE_LOW: <no description>
      CONFIDENCE_MEDIUM: <no description>
      CONFIDENCE_HIGH: <no description>
    """
        CONFIDENCE_UNSPECIFIED = 0
        CONFIDENCE_NONE = 1
        CONFIDENCE_LOW = 2
        CONFIDENCE_MEDIUM = 3
        CONFIDENCE_HIGH = 4

    class RaiCategoryValueValuesEnum(_messages.Enum):
        """The RAI category.

    Values:
      RAI_CATEGORY_UNSPECIFIED: <no description>
      TOXIC: SafetyCat categories.
      SEXUALLY_EXPLICIT: <no description>
      HATE_SPEECH: <no description>
      VIOLENT: <no description>
      PROFANITY: <no description>
      HARASSMENT: <no description>
      DEATH_HARM_TRAGEDY: <no description>
      FIREARMS_WEAPONS: <no description>
      PUBLIC_SAFETY: <no description>
      HEALTH: <no description>
      RELIGIOUS_BELIEF: <no description>
      ILLICIT_DRUGS: <no description>
      WAR_CONFLICT: <no description>
      POLITICS: <no description>
      FINANCE: <no description>
      LEGAL: <no description>
      CSAI: GRAIL categories that can't be exposed to end users.
      FRINGE: <no description>
      THREAT: Unused categories.
      SEVERE_TOXICITY: <no description>
      TOXICITY: Old category names.
      SEXUAL: <no description>
      INSULT: <no description>
      DEROGATORY: <no description>
      IDENTITY_ATTACK: <no description>
      VIOLENCE_ABUSE: <no description>
      OBSCENE: <no description>
      DRUGS: <no description>
      CSAM: CSAM V2
      SPII: SPII
      DANGEROUS_CONTENT: New SafetyCat v3 categories
      DANGEROUS_CONTENT_SEVERITY: <no description>
      INSULT_SEVERITY: <no description>
      DEROGATORY_SEVERITY: <no description>
      SEXUAL_SEVERITY: <no description>
    """
        RAI_CATEGORY_UNSPECIFIED = 0
        TOXIC = 1
        SEXUALLY_EXPLICIT = 2
        HATE_SPEECH = 3
        VIOLENT = 4
        PROFANITY = 5
        HARASSMENT = 6
        DEATH_HARM_TRAGEDY = 7
        FIREARMS_WEAPONS = 8
        PUBLIC_SAFETY = 9
        HEALTH = 10
        RELIGIOUS_BELIEF = 11
        ILLICIT_DRUGS = 12
        WAR_CONFLICT = 13
        POLITICS = 14
        FINANCE = 15
        LEGAL = 16
        CSAI = 17
        FRINGE = 18
        THREAT = 19
        SEVERE_TOXICITY = 20
        TOXICITY = 21
        SEXUAL = 22
        INSULT = 23
        DEROGATORY = 24
        IDENTITY_ATTACK = 25
        VIOLENCE_ABUSE = 26
        OBSCENE = 27
        DRUGS = 28
        CSAM = 29
        SPII = 30
        DANGEROUS_CONTENT = 31
        DANGEROUS_CONTENT_SEVERITY = 32
        INSULT_SEVERITY = 33
        DEROGATORY_SEVERITY = 34
        SEXUAL_SEVERITY = 35
    confidence = _messages.EnumField('ConfidenceValueValuesEnum', 1)
    flagged = _messages.BooleanField(2)
    influentialTerms = _messages.MessageField('CloudAiNlLlmProtoServiceRaiSignalInfluentialTerm', 3, repeated=True)
    raiCategory = _messages.EnumField('RaiCategoryValueValuesEnum', 4)
    score = _messages.FloatField(5, variant=_messages.Variant.FLOAT)