from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RacyValueValuesEnum(_messages.Enum):
    """Likelihood that the request image contains racy content. Racy content
    may include (but is not limited to) skimpy or sheer clothing,
    strategically covered nudity, lewd or provocative poses, or close-ups of
    sensitive body areas.

    Values:
      UNKNOWN: Unknown likelihood.
      VERY_UNLIKELY: It is very unlikely.
      UNLIKELY: It is unlikely.
      POSSIBLE: It is possible.
      LIKELY: It is likely.
      VERY_LIKELY: It is very likely.
    """
    UNKNOWN = 0
    VERY_UNLIKELY = 1
    UNLIKELY = 2
    POSSIBLE = 3
    LIKELY = 4
    VERY_LIKELY = 5