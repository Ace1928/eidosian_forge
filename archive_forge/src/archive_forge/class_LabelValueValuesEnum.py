from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class LabelValueValuesEnum(_messages.Enum):
    """LabelValueValuesEnum enum type.

    Values:
      LABEL_UNSPECIFIED: Default label.
      NOT_SENSITIVE: Input is not sensitive.
      SENSITIVE: Input is sensitive.
      ACCIDENTS_DISASTERS: Input is related to accidents or disasters.
      ADULT: Input contains adult content.
      COMPUTER_SECURITY: Input is related to computer security.
      CONTROVERSIAL_SOCIAL_ISSUES: Input contains controversial social issues.
      DEATH_TRAGEDY: Input is related to death tragedy.
      DRUGS: Input is related to drugs.
      IDENTITY_ETHNICITY: Input is related to identity or ethnicity.
      FINANCIAL_HARDSHIP: Input is related to financial hardship.
      FIREARMS_WEAPONS: Input is related to firearms or weapons.
      HEALTH: Input contains health related information.
      INSULT: Input may be an insult.
      LEGAL: Input is related to legal content.
      MENTAL_HEALTH: Input contains mental health related information.
      POLITICS: Input is related to politics.
      RELIGION_BELIEFS: Input is related to religions or beliefs.
      SAFETY: Input is related to safety.
      SELF_HARM: Input is related to self-harm.
      SPECIAL_NEEDS: Input is related to special needs.
      TERRORISM: Input is related to terrorism.
      TOXIC: Input is toxic.
      TROUBLED_RELATIONSHIP: Input is related to troubled relationships.
      VIOLENCE_ABUSE: Input contains content about violence or abuse.
      VULGAR: Input is vulgar.
      WAR_CONFLICT: Input is related to war and conflict.
    """
    LABEL_UNSPECIFIED = 0
    NOT_SENSITIVE = 1
    SENSITIVE = 2
    ACCIDENTS_DISASTERS = 3
    ADULT = 4
    COMPUTER_SECURITY = 5
    CONTROVERSIAL_SOCIAL_ISSUES = 6
    DEATH_TRAGEDY = 7
    DRUGS = 8
    IDENTITY_ETHNICITY = 9
    FINANCIAL_HARDSHIP = 10
    FIREARMS_WEAPONS = 11
    HEALTH = 12
    INSULT = 13
    LEGAL = 14
    MENTAL_HEALTH = 15
    POLITICS = 16
    RELIGION_BELIEFS = 17
    SAFETY = 18
    SELF_HARM = 19
    SPECIAL_NEEDS = 20
    TERRORISM = 21
    TOXIC = 22
    TROUBLED_RELATIONSHIP = 23
    VIOLENCE_ABUSE = 24
    VULGAR = 25
    WAR_CONFLICT = 26