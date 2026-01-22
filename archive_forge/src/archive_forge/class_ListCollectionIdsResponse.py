from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListCollectionIdsResponse(_messages.Message):
    """The response from Firestore.ListCollectionIds.

  Fields:
    collectionIds: The collection ids.
    nextPageToken: A page token that may be used to continue the list.
  """
    collectionIds = _messages.StringField(1, repeated=True)
    nextPageToken = _messages.StringField(2)