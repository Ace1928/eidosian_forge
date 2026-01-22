from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BulkInsertOperationStatus(_messages.Message):
    """A BulkInsertOperationStatus object.

  Enums:
    StatusValueValuesEnum: [Output Only] Creation status of BulkInsert
      operation - information if the flow is rolling forward or rolling back.

  Fields:
    createdVmCount: [Output Only] Count of VMs successfully created so far.
    deletedVmCount: [Output Only] Count of VMs that got deleted during
      rollback.
    failedToCreateVmCount: [Output Only] Count of VMs that started creating
      but encountered an error.
    status: [Output Only] Creation status of BulkInsert operation -
      information if the flow is rolling forward or rolling back.
    targetVmCount: [Output Only] Count of VMs originally planned to be
      created.
  """

    class StatusValueValuesEnum(_messages.Enum):
        """[Output Only] Creation status of BulkInsert operation - information if
    the flow is rolling forward or rolling back.

    Values:
      STATUS_UNSPECIFIED: <no description>
      CREATING: Rolling forward - creating VMs.
      ROLLING_BACK: Rolling back - cleaning up after an error.
      DONE: Done
    """
        STATUS_UNSPECIFIED = 0
        CREATING = 1
        ROLLING_BACK = 2
        DONE = 3
    createdVmCount = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    deletedVmCount = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    failedToCreateVmCount = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    status = _messages.EnumField('StatusValueValuesEnum', 4)
    targetVmCount = _messages.IntegerField(5, variant=_messages.Variant.INT32)