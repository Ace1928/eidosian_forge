from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class UndeleteOauthClientRequest(_messages.Message):
    """Request message for UndeleteOauthClient.

  Fields:
    validateOnly: Optional. If set, validate the request and preview the
      response, but do not actually post it.
  """
    validateOnly = _messages.BooleanField(1)