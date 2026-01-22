from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PersonValueValuesEnum(_messages.Enum):
    """The grammatical person.

    Values:
      PERSON_UNKNOWN: Person is not applicable in the analyzed language or is
        not predicted.
      FIRST: First
      SECOND: Second
      THIRD: Third
      REFLEXIVE_PERSON: Reflexive
    """
    PERSON_UNKNOWN = 0
    FIRST = 1
    SECOND = 2
    THIRD = 3
    REFLEXIVE_PERSON = 4