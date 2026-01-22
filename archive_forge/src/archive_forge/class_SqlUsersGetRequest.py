from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class SqlUsersGetRequest(_messages.Message):
    """A SqlUsersGetRequest object.

  Fields:
    host: Host of a user of the instance.
    instance: Database instance ID. This does not include the project ID.
    name: User of the instance.
    project: Project ID of the project that contains the instance.
  """
    host = _messages.StringField(1)
    instance = _messages.StringField(2, required=True)
    name = _messages.StringField(3, required=True)
    project = _messages.StringField(4, required=True)