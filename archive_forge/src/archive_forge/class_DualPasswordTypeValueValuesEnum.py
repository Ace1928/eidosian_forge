from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class DualPasswordTypeValueValuesEnum(_messages.Enum):
    """Dual password status for the user.

    Values:
      DUAL_PASSWORD_TYPE_UNSPECIFIED: The default value.
      NO_MODIFY_DUAL_PASSWORD: Do not update the user's dual password status.
      NO_DUAL_PASSWORD: No dual password usable for connecting using this
        user.
      DUAL_PASSWORD: Dual password usable for connecting using this user.
    """
    DUAL_PASSWORD_TYPE_UNSPECIFIED = 0
    NO_MODIFY_DUAL_PASSWORD = 1
    NO_DUAL_PASSWORD = 2
    DUAL_PASSWORD = 3