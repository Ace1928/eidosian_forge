from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowCxV3beta1WebhookRequestIntentInfoIntentParameterValue(_messages.Message):
    """Represents a value for an intent parameter.

  Fields:
    originalValue: Always present. Original text value extracted from user
      utterance.
    resolvedValue: Always present. Structured value for the parameter
      extracted from user utterance.
  """
    originalValue = _messages.StringField(1)
    resolvedValue = _messages.MessageField('extra_types.JsonValue', 2)