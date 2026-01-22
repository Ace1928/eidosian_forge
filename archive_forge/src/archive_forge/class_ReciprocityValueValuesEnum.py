from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ReciprocityValueValuesEnum(_messages.Enum):
    """The grammatical reciprocity.

    Values:
      RECIPROCITY_UNKNOWN: Reciprocity is not applicable in the analyzed
        language or is not predicted.
      RECIPROCAL: Reciprocal
      NON_RECIPROCAL: Non-reciprocal
    """
    RECIPROCITY_UNKNOWN = 0
    RECIPROCAL = 1
    NON_RECIPROCAL = 2