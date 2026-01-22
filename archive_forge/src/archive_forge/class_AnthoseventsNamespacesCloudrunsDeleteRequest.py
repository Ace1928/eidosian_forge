from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class AnthoseventsNamespacesCloudrunsDeleteRequest(_messages.Message):
    """A AnthoseventsNamespacesCloudrunsDeleteRequest object.

  Fields:
    name: The name of the CloudRun resource being deleted.
  """
    name = _messages.StringField(1, required=True)