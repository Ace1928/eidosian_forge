from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class AnthoseventsNamespacesTriggersCreateRequest(_messages.Message):
    """A AnthoseventsNamespacesTriggersCreateRequest object.

  Fields:
    parent: The namespace name.
    trigger: A Trigger resource to be passed as the request body.
  """
    parent = _messages.StringField(1, required=True)
    trigger = _messages.MessageField('Trigger', 2)