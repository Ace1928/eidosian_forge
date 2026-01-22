from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListBuildTriggersResponse(_messages.Message):
    """Response containing existing `BuildTriggers`.

  Fields:
    nextPageToken: Token to receive the next page of results.
    triggers: `BuildTriggers` for the project, sorted by `create_time`
      descending.
  """
    nextPageToken = _messages.StringField(1)
    triggers = _messages.MessageField('BuildTrigger', 2, repeated=True)