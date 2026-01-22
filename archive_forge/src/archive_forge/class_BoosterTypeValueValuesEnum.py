from __future__ import absolute_import
from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BoosterTypeValueValuesEnum(_messages.Enum):
    """Booster type for boosted tree models.

    Values:
      BOOSTER_TYPE_UNSPECIFIED: Unspecified booster type.
      GBTREE: Gbtree booster.
      DART: Dart booster.
    """
    BOOSTER_TYPE_UNSPECIFIED = 0
    GBTREE = 1
    DART = 2