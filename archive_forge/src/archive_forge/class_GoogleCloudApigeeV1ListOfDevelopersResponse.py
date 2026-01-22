from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudApigeeV1ListOfDevelopersResponse(_messages.Message):
    """A GoogleCloudApigeeV1ListOfDevelopersResponse object.

  Fields:
    developer: List of developers.
    nextPageToken: Token that can be sent as `next_page_token` to retrieve the
      next page. If this field is omitted, there are no subsequent pages.
    totalSize: Total count of Developers.
  """
    developer = _messages.MessageField('GoogleCloudApigeeV1Developer', 1, repeated=True)
    nextPageToken = _messages.StringField(2)
    totalSize = _messages.IntegerField(3, variant=_messages.Variant.INT32)