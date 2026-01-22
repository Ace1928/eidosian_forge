from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudresourcemanagerProjectsMoveRequest(_messages.Message):
    """A CloudresourcemanagerProjectsMoveRequest object.

  Fields:
    moveProjectRequest: A MoveProjectRequest resource to be passed as the
      request body.
    name: Required. The name of the project to move.
  """
    moveProjectRequest = _messages.MessageField('MoveProjectRequest', 1)
    name = _messages.StringField(2, required=True)