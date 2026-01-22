from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SpannerProjectsInstancesDatabasesCreateRequest(_messages.Message):
    """A SpannerProjectsInstancesDatabasesCreateRequest object.

  Fields:
    createDatabaseRequest: A CreateDatabaseRequest resource to be passed as
      the request body.
    parent: Required. The name of the instance that will serve the new
      database. Values are of the form `projects//instances/`.
  """
    createDatabaseRequest = _messages.MessageField('CreateDatabaseRequest', 1)
    parent = _messages.StringField(2, required=True)