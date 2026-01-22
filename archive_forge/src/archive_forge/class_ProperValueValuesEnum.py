from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ProperValueValuesEnum(_messages.Enum):
    """The grammatical properness.

    Values:
      PROPER_UNKNOWN: Proper is not applicable in the analyzed language or is
        not predicted.
      PROPER: Proper
      NOT_PROPER: Not proper
    """
    PROPER_UNKNOWN = 0
    PROPER = 1
    NOT_PROPER = 2