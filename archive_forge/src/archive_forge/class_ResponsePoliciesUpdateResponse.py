from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class ResponsePoliciesUpdateResponse(_messages.Message):
    """A ResponsePoliciesUpdateResponse object.

  Fields:
    header: A ResponseHeader attribute.
    responsePolicy: A ResponsePolicy attribute.
  """
    header = _messages.MessageField('ResponseHeader', 1)
    responsePolicy = _messages.MessageField('ResponsePolicy', 2)