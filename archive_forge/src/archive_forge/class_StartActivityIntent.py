from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class StartActivityIntent(_messages.Message):
    """A starting intent specified by an action, uri, and categories.

  Fields:
    action: Action name. Required for START_ACTIVITY.
    categories: Intent categories to set on the intent.
    uri: URI for the action.
  """
    action = _messages.StringField(1)
    categories = _messages.StringField(2, repeated=True)
    uri = _messages.StringField(3)