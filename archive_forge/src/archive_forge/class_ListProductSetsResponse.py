from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListProductSetsResponse(_messages.Message):
    """Response message for the `ListProductSets` method.

  Fields:
    nextPageToken: Token to retrieve the next page of results, or empty if
      there are no more results in the list.
    productSets: List of ProductSets.
  """
    nextPageToken = _messages.StringField(1)
    productSets = _messages.MessageField('ProductSet', 2, repeated=True)