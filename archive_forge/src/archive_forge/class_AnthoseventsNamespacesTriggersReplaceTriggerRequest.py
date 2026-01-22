from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class AnthoseventsNamespacesTriggersReplaceTriggerRequest(_messages.Message):
    """A AnthoseventsNamespacesTriggersReplaceTriggerRequest object.

  Fields:
    name: The name of the trigger being retrieved. If needed, replace
      {namespace_id} with the project ID.
    trigger: A Trigger resource to be passed as the request body.
  """
    name = _messages.StringField(1, required=True)
    trigger = _messages.MessageField('Trigger', 2)