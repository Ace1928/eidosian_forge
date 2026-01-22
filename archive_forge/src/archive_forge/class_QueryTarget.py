from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class QueryTarget(_messages.Message):
    """A target specified by a query.

  Fields:
    parent: The parent resource name. In the format:
      `projects/{project_id}/databases/{database_id}/documents` or `projects/{
      project_id}/databases/{database_id}/documents/{document_path}`. For
      example: `projects/my-project/databases/my-database/documents` or
      `projects/my-project/databases/my-database/documents/chatrooms/my-
      chatroom`
    structuredQuery: A structured query.
  """
    parent = _messages.StringField(1)
    structuredQuery = _messages.MessageField('StructuredQuery', 2)