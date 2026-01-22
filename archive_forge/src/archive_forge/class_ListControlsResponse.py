from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListControlsResponse(_messages.Message):
    """Response message with all the controls for a compliance standard.

  Fields:
    controls: Output only. The controls for the compliance standard.
    nextPageToken: Output only. The token to retrieve the next page of
      results.
  """
    controls = _messages.MessageField('Control', 1, repeated=True)
    nextPageToken = _messages.StringField(2)