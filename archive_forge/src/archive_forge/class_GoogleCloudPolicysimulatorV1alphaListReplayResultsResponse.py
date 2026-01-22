from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudPolicysimulatorV1alphaListReplayResultsResponse(_messages.Message):
    """Response message for Simulator.ListReplayResults.

  Fields:
    nextPageToken: A token that you can use to retrieve the next page of
      ReplayResult objects. If this field is omitted, there are no subsequent
      pages.
    replayResults: The results of running a Replay.
  """
    nextPageToken = _messages.StringField(1)
    replayResults = _messages.MessageField('GoogleCloudPolicysimulatorV1alphaReplayResult', 2, repeated=True)