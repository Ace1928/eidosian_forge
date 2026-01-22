from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudApigeeV1ListApiDocsResponse(_messages.Message):
    """A GoogleCloudApigeeV1ListApiDocsResponse object.

  Fields:
    data: The catalog item resources.
    errorCode: Unique error code for the request, if any.
    message: Description of the operation.
    nextPageToken: A token, which can be sent as `page_token` to retrieve the
      next page. If this field is omitted, there are no subsequent pages.
    requestId: Unique ID of the request.
    status: Status of the operation.
  """
    data = _messages.MessageField('GoogleCloudApigeeV1ApiDoc', 1, repeated=True)
    errorCode = _messages.StringField(2)
    message = _messages.StringField(3)
    nextPageToken = _messages.StringField(4)
    requestId = _messages.StringField(5)
    status = _messages.StringField(6)