from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudApigeeV1DeleteResponse(_messages.Message):
    """Response for certain delete operations.

  Fields:
    errorCode: Unique error code for the request, if any.
    gcpResource: Google Cloud name of deleted resource.
    message: Description of the operation.
    requestId: Unique ID of the request.
    status: Status of the operation.
  """
    errorCode = _messages.StringField(1)
    gcpResource = _messages.StringField(2)
    message = _messages.StringField(3)
    requestId = _messages.StringField(4)
    status = _messages.StringField(5)