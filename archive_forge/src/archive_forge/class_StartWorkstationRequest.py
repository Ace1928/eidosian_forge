from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class StartWorkstationRequest(_messages.Message):
    """Request message for StartWorkstation.

  Fields:
    boostConfig: Optional. If set, the workstation starts using the boost
      configuration with the specified ID.
    etag: Optional. If set, the request will be rejected if the latest version
      of the workstation on the server does not have this ETag.
    validateOnly: Optional. If set, validate the request and preview the
      review, but do not actually apply it.
  """
    boostConfig = _messages.StringField(1)
    etag = _messages.StringField(2)
    validateOnly = _messages.BooleanField(3)