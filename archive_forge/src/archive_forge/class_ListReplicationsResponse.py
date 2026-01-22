from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListReplicationsResponse(_messages.Message):
    """ListReplicationsResponse is the result of ListReplicationsRequest.

  Fields:
    nextPageToken: The token you can use to retrieve the next page of results.
      Not returned if there are no more results in the list.
    replications: A list of replications in the project for the specified
      volume.
    unreachable: Locations that could not be reached.
  """
    nextPageToken = _messages.StringField(1)
    replications = _messages.MessageField('Replication', 2, repeated=True)
    unreachable = _messages.StringField(3, repeated=True)