from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class UpdateInstancePartitionRequest(_messages.Message):
    """The request for UpdateInstancePartition.

  Fields:
    fieldMask: Required. A mask specifying which fields in InstancePartition
      should be updated. The field mask must always be specified; this
      prevents any future fields in InstancePartition from being erased
      accidentally by clients that do not know about them.
    instancePartition: Required. The instance partition to update, which must
      always include the instance partition name. Otherwise, only fields
      mentioned in field_mask need be included.
  """
    fieldMask = _messages.StringField(1)
    instancePartition = _messages.MessageField('InstancePartition', 2)