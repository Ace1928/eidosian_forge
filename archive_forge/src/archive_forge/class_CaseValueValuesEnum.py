from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CaseValueValuesEnum(_messages.Enum):
    """The grammatical case.

    Values:
      CASE_UNKNOWN: Case is not applicable in the analyzed language or is not
        predicted.
      ACCUSATIVE: Accusative
      ADVERBIAL: Adverbial
      COMPLEMENTIVE: Complementive
      DATIVE: Dative
      GENITIVE: Genitive
      INSTRUMENTAL: Instrumental
      LOCATIVE: Locative
      NOMINATIVE: Nominative
      OBLIQUE: Oblique
      PARTITIVE: Partitive
      PREPOSITIONAL: Prepositional
      REFLEXIVE_CASE: Reflexive
      RELATIVE_CASE: Relative
      VOCATIVE: Vocative
    """
    CASE_UNKNOWN = 0
    ACCUSATIVE = 1
    ADVERBIAL = 2
    COMPLEMENTIVE = 3
    DATIVE = 4
    GENITIVE = 5
    INSTRUMENTAL = 6
    LOCATIVE = 7
    NOMINATIVE = 8
    OBLIQUE = 9
    PARTITIVE = 10
    PREPOSITIONAL = 11
    REFLEXIVE_CASE = 12
    RELATIVE_CASE = 13
    VOCATIVE = 14