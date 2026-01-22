from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class FirestoreProjectsDatabasesCollectionGroupsIndexesListRequest(_messages.Message):
    """A FirestoreProjectsDatabasesCollectionGroupsIndexesListRequest object.

  Fields:
    filter: The filter to apply to list results.
    pageSize: The number of results to return.
    pageToken: A page token, returned from a previous call to
      FirestoreAdmin.ListIndexes, that may be used to get the next page of
      results.
    parent: Required. A parent name of the form `projects/{project_id}/databas
      es/{database_id}/collectionGroups/{collection_id}`
  """
    filter = _messages.StringField(1)
    pageSize = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(3)
    parent = _messages.StringField(4, required=True)