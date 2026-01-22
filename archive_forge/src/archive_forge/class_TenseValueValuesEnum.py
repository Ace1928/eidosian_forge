from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TenseValueValuesEnum(_messages.Enum):
    """The grammatical tense.

    Values:
      TENSE_UNKNOWN: Tense is not applicable in the analyzed language or is
        not predicted.
      CONDITIONAL_TENSE: Conditional
      FUTURE: Future
      PAST: Past
      PRESENT: Present
      IMPERFECT: Imperfect
      PLUPERFECT: Pluperfect
    """
    TENSE_UNKNOWN = 0
    CONDITIONAL_TENSE = 1
    FUTURE = 2
    PAST = 3
    PRESENT = 4
    IMPERFECT = 5
    PLUPERFECT = 6