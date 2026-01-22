from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PairwiseChoiceValueValuesEnum(_messages.Enum):
    """Output only. Pairwise summarization prediction choice.

    Values:
      PAIRWISE_CHOICE_UNSPECIFIED: Unspecified prediction choice.
      BASELINE: Baseline prediction wins
      CANDIDATE: Candidate prediction wins
      TIE: Winner cannot be determined
    """
    PAIRWISE_CHOICE_UNSPECIFIED = 0
    BASELINE = 1
    CANDIDATE = 2
    TIE = 3