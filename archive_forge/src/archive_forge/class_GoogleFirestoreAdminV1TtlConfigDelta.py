from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleFirestoreAdminV1TtlConfigDelta(_messages.Message):
    """Information about a TTL configuration change.

  Enums:
    ChangeTypeValueValuesEnum: Specifies how the TTL configuration is
      changing.

  Fields:
    changeType: Specifies how the TTL configuration is changing.
  """

    class ChangeTypeValueValuesEnum(_messages.Enum):
        """Specifies how the TTL configuration is changing.

    Values:
      CHANGE_TYPE_UNSPECIFIED: The type of change is not specified or known.
      ADD: The TTL config is being added.
      REMOVE: The TTL config is being removed.
    """
        CHANGE_TYPE_UNSPECIFIED = 0
        ADD = 1
        REMOVE = 2
    changeType = _messages.EnumField('ChangeTypeValueValuesEnum', 1)