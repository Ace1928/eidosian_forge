from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListServiceAccountIdentityBindingsResponse(_messages.Message):
    """The service account identity bindings list response.

  Fields:
    identityBinding: The identity bindings trusted to assert the service
      account.
  """
    identityBinding = _messages.MessageField('ServiceAccountIdentityBinding', 1, repeated=True)