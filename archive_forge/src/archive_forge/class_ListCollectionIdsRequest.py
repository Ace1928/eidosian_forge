from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListCollectionIdsRequest(_messages.Message):
    """The request for Firestore.ListCollectionIds.

  Fields:
    pageSize: The maximum number of results to return.
    pageToken: A page token. Must be a value from ListCollectionIdsResponse.
    readTime: Reads documents as they were at the given time. This must be a
      microsecond precision timestamp within the past one hour, or if Point-
      in-Time Recovery is enabled, can additionally be a whole minute
      timestamp within the past 7 days.
  """
    pageSize = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(2)
    readTime = _messages.StringField(3)