from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowCxV3beta1IntentInput(_messages.Message):
    """Represents the intent to trigger programmatically rather than as a
  result of natural language processing.

  Fields:
    intent: Required. The unique identifier of the intent. Format:
      `projects//locations//agents//intents/`.
  """
    intent = _messages.StringField(1)