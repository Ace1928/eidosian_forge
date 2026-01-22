from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListServiceApisResponse(_messages.Message):
    """The response message of the `ListServiceApis` method.

  Fields:
    apis: The apis exposed by the parent service.
    nextPageToken: A token, which can be sent as `page_token` to retrieve the
      next page. If this field is omitted, there are no subsequent pages.
  """
    apis = _messages.MessageField('GoogleApiServiceusageV2alphaApi', 1, repeated=True)
    nextPageToken = _messages.StringField(2)