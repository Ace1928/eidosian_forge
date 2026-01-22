from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class GetRevisionsResponse(_messages.Message):
    """Response for GetRevisions.

  Fields:
    revisions: The revisions.
  """
    revisions = _messages.MessageField('Revision', 1, repeated=True)