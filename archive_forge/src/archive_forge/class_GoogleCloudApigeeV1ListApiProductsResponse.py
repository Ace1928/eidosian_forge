from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudApigeeV1ListApiProductsResponse(_messages.Message):
    """A GoogleCloudApigeeV1ListApiProductsResponse object.

  Fields:
    apiProduct: Lists all API product names defined for an organization.
    nextPageToken: Token that can be sent as `next_page_token` to retrieve the
      next page. If this field is omitted, there are no subsequent pages.
    totalSize: Total count of API products for this org.
  """
    apiProduct = _messages.MessageField('GoogleCloudApigeeV1ApiProduct', 1, repeated=True)
    nextPageToken = _messages.StringField(2)
    totalSize = _messages.IntegerField(3, variant=_messages.Variant.INT32)