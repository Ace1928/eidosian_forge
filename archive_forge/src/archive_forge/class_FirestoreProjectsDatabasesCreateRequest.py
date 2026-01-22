from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class FirestoreProjectsDatabasesCreateRequest(_messages.Message):
    """A FirestoreProjectsDatabasesCreateRequest object.

  Fields:
    databaseId: Required. The ID to use for the database, which will become
      the final component of the database's resource name. This value should
      be 4-63 characters. Valid characters are /a-z-/ with first character a
      letter and the last a letter or a number. Must not be UUID-like
      /[0-9a-f]{8}(-[0-9a-f]{4}){3}-[0-9a-f]{12}/. "(default)" database id is
      also valid.
    googleFirestoreAdminV1Database: A GoogleFirestoreAdminV1Database resource
      to be passed as the request body.
    parent: Required. A parent name of the form `projects/{project_id}`
  """
    databaseId = _messages.StringField(1)
    googleFirestoreAdminV1Database = _messages.MessageField('GoogleFirestoreAdminV1Database', 2)
    parent = _messages.StringField(3, required=True)