from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowCxV3beta1ResponseMessageMixedAudioSegment(_messages.Message):
    """Represents one segment of audio.

  Fields:
    allowPlaybackInterruption: Output only. Whether the playback of this
      segment can be interrupted by the end user's speech and the client
      should then start the next Dialogflow request.
    audio: Raw audio synthesized from the Dialogflow agent's response using
      the output config specified in the request.
    uri: Client-specific URI that points to an audio clip accessible to the
      client. Dialogflow does not impose any validation on it.
  """
    allowPlaybackInterruption = _messages.BooleanField(1)
    audio = _messages.BytesField(2)
    uri = _messages.StringField(3)