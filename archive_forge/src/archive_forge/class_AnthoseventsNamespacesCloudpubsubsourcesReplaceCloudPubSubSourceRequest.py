from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class AnthoseventsNamespacesCloudpubsubsourcesReplaceCloudPubSubSourceRequest(_messages.Message):
    """A
  AnthoseventsNamespacesCloudpubsubsourcesReplaceCloudPubSubSourceRequest
  object.

  Fields:
    cloudPubSubSource: A CloudPubSubSource resource to be passed as the
      request body.
    name: The name of the cloudpubsubsource being retrieved. If needed,
      replace {namespace_id} with the project ID.
  """
    cloudPubSubSource = _messages.MessageField('CloudPubSubSource', 1)
    name = _messages.StringField(2, required=True)