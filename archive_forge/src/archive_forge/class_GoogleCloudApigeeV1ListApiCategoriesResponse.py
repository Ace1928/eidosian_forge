from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudApigeeV1ListApiCategoriesResponse(_messages.Message):
    """The response for `ListApiCategoriesRequest`. Next ID: 6

  Fields:
    data: The API category resources.
    errorCode: Unique error code for the request, if any.
    message: Description of the operation.
    requestId: Unique ID of the request.
    status: Status of the operation.
  """
    data = _messages.MessageField('GoogleCloudApigeeV1ApiCategory', 1, repeated=True)
    errorCode = _messages.StringField(2)
    message = _messages.StringField(3)
    requestId = _messages.StringField(4)
    status = _messages.StringField(5)