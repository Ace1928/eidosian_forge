from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DynamicGroupStatus(_messages.Message):
    """The current status of a dynamic group along with timestamp.

  Enums:
    StatusValueValuesEnum: Status of the dynamic group.

  Fields:
    status: Status of the dynamic group.
    statusTime: The latest time at which the dynamic group is guaranteed to be
      in the given status. If status is `UP_TO_DATE`, the latest time at which
      the dynamic group was confirmed to be up-to-date. If status is
      `UPDATING_MEMBERSHIPS`, the time at which dynamic group was created.
  """

    class StatusValueValuesEnum(_messages.Enum):
        """Status of the dynamic group.

    Values:
      STATUS_UNSPECIFIED: Default.
      UP_TO_DATE: The dynamic group is up-to-date.
      UPDATING_MEMBERSHIPS: The dynamic group has just been created and
        memberships are being updated.
      INVALID_QUERY: Group is in an unrecoverable state and its memberships
        can't be updated.
    """
        STATUS_UNSPECIFIED = 0
        UP_TO_DATE = 1
        UPDATING_MEMBERSHIPS = 2
        INVALID_QUERY = 3
    status = _messages.EnumField('StatusValueValuesEnum', 1)
    statusTime = _messages.StringField(2)