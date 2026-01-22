from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PolicysimulatorProjectsLocationsReplaysResultsListRequest(_messages.Message):
    """A PolicysimulatorProjectsLocationsReplaysResultsListRequest object.

  Fields:
    pageSize: The maximum number of ReplayResult objects to return. Defaults
      to 5000. The maximum value is 5000; values above 5000 are rounded down
      to 5000.
    pageToken: A page token, received from a previous
      Simulator.ListReplayResults call. Provide this token to retrieve the
      next page of results. When paginating, all other parameters provided to
      [Simulator.ListReplayResults[] must match the call that provided the
      page token.
    parent: Required. The Replay whose results are listed, in the following
      format: `{projects|folders|organizations}/{resource-
      id}/locations/global/replays/{replay-id}` Example: `projects/my-
      project/locations/global/replays/506a5f7f-38ce-4d7d-8e03-479ce1833c36`
  """
    pageSize = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)