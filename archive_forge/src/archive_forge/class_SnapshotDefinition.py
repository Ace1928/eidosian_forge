from __future__ import absolute_import
from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SnapshotDefinition(_messages.Message):
    """Information about base table and snapshot time of the snapshot.

  Fields:
    baseTableReference: Required. Reference describing the ID of the table
      that was snapshot.
    snapshotTime: Required. The time at which the base table was snapshot.
      This value is reported in the JSON response using RFC3339 format.
  """
    baseTableReference = _messages.MessageField('TableReference', 1)
    snapshotTime = _message_types.DateTimeField(2)