from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class AnthoseventsNamespacesCloudrunsGetRequest(_messages.Message):
    """A AnthoseventsNamespacesCloudrunsGetRequest object.

  Fields:
    name: The name of the CloudRun resource being retrieved.
  """
    name = _messages.StringField(1, required=True)