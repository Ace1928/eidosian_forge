from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CreateOperationRequest(_messages.Message):
    """Request for creating an operation

  Fields:
    operation: The operation to create.
    operationId: The ID to use for this operation.
  """
    operation = _messages.MessageField('Operation', 1)
    operationId = _messages.StringField(2)