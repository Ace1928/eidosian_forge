from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class ListSkusResponse(_messages.Message):
    """Response message for `ListSkus`.

  Fields:
    nextPageToken: A token to retrieve the next page of results. To retrieve
      the next page, call `ListSkus` again with the `page_token` field set to
      this value. This field is empty if there are no more results to
      retrieve.
    skus: The list of public SKUs of the given service.
  """
    nextPageToken = _messages.StringField(1)
    skus = _messages.MessageField('Sku', 2, repeated=True)