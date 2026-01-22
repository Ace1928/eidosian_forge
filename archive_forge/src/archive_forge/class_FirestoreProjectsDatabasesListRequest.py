from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class FirestoreProjectsDatabasesListRequest(_messages.Message):
    """A FirestoreProjectsDatabasesListRequest object.

  Fields:
    parent: Required. A parent name of the form `projects/{project_id}`
  """
    parent = _messages.StringField(1, required=True)