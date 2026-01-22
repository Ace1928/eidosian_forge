from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class V1Beta1GenerateServiceIdentityResponse(_messages.Message):
    """Response message for the `GenerateServiceIdentity` method.  This
  response message is assigned to the `response` field of the returned
  Operation when that operation is done.

  Fields:
    identity: ServiceIdentity that was created or retrieved.
  """
    identity = _messages.MessageField('V1Beta1ServiceIdentity', 1)