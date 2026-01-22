from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class FirestoreProjectsDatabasesIndexesDeleteRequest(_messages.Message):
    """A FirestoreProjectsDatabasesIndexesDeleteRequest object.

  Fields:
    name: The index name. For example:
      `projects/{project_id}/databases/{database_id}/indexes/{index_id}`
  """
    name = _messages.StringField(1, required=True)