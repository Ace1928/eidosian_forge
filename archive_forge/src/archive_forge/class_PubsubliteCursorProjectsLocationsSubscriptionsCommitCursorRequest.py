from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PubsubliteCursorProjectsLocationsSubscriptionsCommitCursorRequest(_messages.Message):
    """A PubsubliteCursorProjectsLocationsSubscriptionsCommitCursorRequest
  object.

  Fields:
    commitCursorRequest: A CommitCursorRequest resource to be passed as the
      request body.
    subscription: The subscription for which to update the cursor.
  """
    commitCursorRequest = _messages.MessageField('CommitCursorRequest', 1)
    subscription = _messages.StringField(2, required=True)