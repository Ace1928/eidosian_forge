from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class ListCloudPubSubSourcesResponse(_messages.Message):
    """ListCloudPubSubSourcesResponse is a list of CloudPubSubSource resources.

  Fields:
    apiVersion: The API version for this call such as
      "events.cloud.google.com/v1".
    items: List of CloudPubSubSources.
    kind: The kind of this resource, in this case "CloudPubSubSourceList".
    metadata: Metadata associated with this CloudPubSubSource list.
    unreachable: Locations that could not be reached.
  """
    apiVersion = _messages.StringField(1)
    items = _messages.MessageField('CloudPubSubSource', 2, repeated=True)
    kind = _messages.StringField(3)
    metadata = _messages.MessageField('ListMeta', 4)
    unreachable = _messages.StringField(5, repeated=True)