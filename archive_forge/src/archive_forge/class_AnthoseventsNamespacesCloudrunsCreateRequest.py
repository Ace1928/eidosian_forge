from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class AnthoseventsNamespacesCloudrunsCreateRequest(_messages.Message):
    """A AnthoseventsNamespacesCloudrunsCreateRequest object.

  Fields:
    cloudRun: A CloudRun resource to be passed as the request body.
    parent: The namespace in which this CloudRun resource should be created.
  """
    cloudRun = _messages.MessageField('CloudRun', 1)
    parent = _messages.StringField(2, required=True)