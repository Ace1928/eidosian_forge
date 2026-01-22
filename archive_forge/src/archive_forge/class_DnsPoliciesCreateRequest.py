from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class DnsPoliciesCreateRequest(_messages.Message):
    """A DnsPoliciesCreateRequest object.

  Fields:
    clientOperationId: For mutating operation requests only. An optional
      identifier specified by the client. Must be unique for operation
      resources in the Operations collection.
    policy: A Policy resource to be passed as the request body.
    project: Identifies the project addressed by this request.
  """
    clientOperationId = _messages.StringField(1)
    policy = _messages.MessageField('Policy', 2)
    project = _messages.StringField(3, required=True)