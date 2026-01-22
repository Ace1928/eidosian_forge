from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class InteractionTypeValueValuesEnum(_messages.Enum):
    """The use case most closely describing the audio content to be
    recognized.

    Values:
      INTERACTION_TYPE_UNSPECIFIED: Use case is either unknown or is something
        other than one of the other values below.
      DISCUSSION: Multiple people in a conversation or discussion. For example
        in a meeting with two or more people actively participating. Typically
        all the primary people speaking would be in the same room (if not, see
        PHONE_CALL)
      PRESENTATION: One or more persons lecturing or presenting to others,
        mostly uninterrupted.
      PHONE_CALL: A phone-call or video-conference in which two or more
        people, who are not in the same room, are actively participating.
      VOICEMAIL: A recorded message intended for another person to listen to.
      PROFESSIONALLY_PRODUCED: Professionally produced audio (eg. TV Show,
        Podcast).
      VOICE_SEARCH: Transcribe spoken questions and queries into text.
      VOICE_COMMAND: Transcribe voice commands, such as for controlling a
        device.
      DICTATION: Transcribe speech to text to create a written document, such
        as a text-message, email or report.
    """
    INTERACTION_TYPE_UNSPECIFIED = 0
    DISCUSSION = 1
    PRESENTATION = 2
    PHONE_CALL = 3
    VOICEMAIL = 4
    PROFESSIONALLY_PRODUCED = 5
    VOICE_SEARCH = 6
    VOICE_COMMAND = 7
    DICTATION = 8