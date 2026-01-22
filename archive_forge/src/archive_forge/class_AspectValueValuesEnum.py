from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AspectValueValuesEnum(_messages.Enum):
    """The grammatical aspect.

    Values:
      ASPECT_UNKNOWN: Aspect is not applicable in the analyzed language or is
        not predicted.
      PERFECTIVE: Perfective
      IMPERFECTIVE: Imperfective
      PROGRESSIVE: Progressive
    """
    ASPECT_UNKNOWN = 0
    PERFECTIVE = 1
    IMPERFECTIVE = 2
    PROGRESSIVE = 3