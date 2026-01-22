from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListPostureRevisionsResponse(_messages.Message):
    """Message for response to listing PostureRevisions.

  Fields:
    nextPageToken: A token identifying a page of results the server should
      return.
    revisions: The list of Posture revisions.
  """
    nextPageToken = _messages.StringField(1)
    revisions = _messages.MessageField('Posture', 2, repeated=True)