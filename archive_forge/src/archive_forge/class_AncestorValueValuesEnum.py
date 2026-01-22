from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AncestorValueValuesEnum(_messages.Enum):
    """Required. The index's ancestor mode. Must not be
    ANCESTOR_MODE_UNSPECIFIED.

    Values:
      ANCESTOR_MODE_UNSPECIFIED: The ancestor mode is unspecified.
      NONE: Do not include the entity's ancestors in the index.
      ALL_ANCESTORS: Include all the entity's ancestors in the index.
    """
    ANCESTOR_MODE_UNSPECIFIED = 0
    NONE = 1
    ALL_ANCESTORS = 2