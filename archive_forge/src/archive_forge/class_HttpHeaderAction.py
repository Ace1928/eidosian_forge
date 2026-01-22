from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class HttpHeaderAction(_messages.Message):
    """The request and response header transformations that take effect before
  the request is passed along to the selected backendService.

  Fields:
    requestHeadersToAdd: Headers to add to a matching request before
      forwarding the request to the backendService.
    requestHeadersToRemove: A list of header names for headers that need to be
      removed from the request before forwarding the request to the
      backendService.
    responseHeadersToAdd: Headers to add the response before sending the
      response back to the client.
    responseHeadersToRemove: A list of header names for headers that need to
      be removed from the response before sending the response back to the
      client.
  """
    requestHeadersToAdd = _messages.MessageField('HttpHeaderOption', 1, repeated=True)
    requestHeadersToRemove = _messages.StringField(2, repeated=True)
    responseHeadersToAdd = _messages.MessageField('HttpHeaderOption', 3, repeated=True)
    responseHeadersToRemove = _messages.StringField(4, repeated=True)