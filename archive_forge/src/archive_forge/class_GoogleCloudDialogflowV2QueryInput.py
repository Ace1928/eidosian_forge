from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowV2QueryInput(_messages.Message):
    """Represents the query input. It can contain either: 1. An audio config
  which instructs the speech recognizer how to process the speech audio. 2. A
  conversational query in the form of text. 3. An event that specifies which
  intent to trigger.

  Fields:
    audioConfig: Instructs the speech recognizer how to process the speech
      audio.
    event: The event to be processed.
    text: The natural language text to be processed. Text length must not
      exceed 256 character for virtual agent interactions.
  """
    audioConfig = _messages.MessageField('GoogleCloudDialogflowV2InputAudioConfig', 1)
    event = _messages.MessageField('GoogleCloudDialogflowV2EventInput', 2)
    text = _messages.MessageField('GoogleCloudDialogflowV2TextInput', 3)