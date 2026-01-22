from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListStreamInstancesResponse(_messages.Message):
    """Message for response to listing StreamInstances

  Fields:
    nextPageToken: A token identifying a page of results the server should
      return.
    streamInstances: The list of StreamInstance
    unreachable: Locations that could not be reached.
  """
    nextPageToken = _messages.StringField(1)
    streamInstances = _messages.MessageField('StreamInstance', 2, repeated=True)
    unreachable = _messages.StringField(3, repeated=True)