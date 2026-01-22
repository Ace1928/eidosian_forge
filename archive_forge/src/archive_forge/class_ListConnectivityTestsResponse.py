from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListConnectivityTestsResponse(_messages.Message):
    """Response for the `ListConnectivityTests` method.

  Fields:
    nextPageToken: Page token to fetch the next set of Connectivity Tests.
    resources: List of Connectivity Tests.
    unreachable: Locations that could not be reached (when querying all
      locations with `-`).
  """
    nextPageToken = _messages.StringField(1)
    resources = _messages.MessageField('ConnectivityTest', 2, repeated=True)
    unreachable = _messages.StringField(3, repeated=True)