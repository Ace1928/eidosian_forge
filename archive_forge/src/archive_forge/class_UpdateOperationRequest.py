from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class UpdateOperationRequest(_messages.Message):
    """Request for updating an existing operation

  Fields:
    operation: The operation to create.
    updateMask: The fields to update.
  """
    operation = _messages.MessageField('Operation', 1)
    updateMask = _messages.StringField(2)