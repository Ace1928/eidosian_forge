from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RecognitionMetadata(_messages.Message):
    """Description of audio data to be recognized.

  Enums:
    InteractionTypeValueValuesEnum: The use case most closely describing the
      audio content to be recognized.
    MicrophoneDistanceValueValuesEnum: The audio type that most closely
      describes the audio being recognized.
    OriginalMediaTypeValueValuesEnum: The original media the speech was
      recorded on.
    RecordingDeviceTypeValueValuesEnum: The type of device the speech was
      recorded with.

  Fields:
    audioTopic: Description of the content. Eg. "Recordings of federal supreme
      court hearings from 2012".
    industryNaicsCodeOfAudio: The industry vertical to which this speech
      recognition request most closely applies. This is most indicative of the
      topics contained in the audio. Use the 6-digit NAICS code to identify
      the industry vertical - see https://www.naics.com/search/.
    interactionType: The use case most closely describing the audio content to
      be recognized.
    microphoneDistance: The audio type that most closely describes the audio
      being recognized.
    originalMediaType: The original media the speech was recorded on.
    originalMimeType: Mime type of the original audio file. For example
      `audio/m4a`, `audio/x-alaw-basic`, `audio/mp3`, `audio/3gpp`. A list of
      possible audio mime types is maintained at
      http://www.iana.org/assignments/media-types/media-types.xhtml#audio
    recordingDeviceName: The device used to make the recording. Examples
      'Nexus 5X' or 'Polycom SoundStation IP 6000' or 'POTS' or 'VoIP' or
      'Cardioid Microphone'.
    recordingDeviceType: The type of device the speech was recorded with.
  """

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

    class MicrophoneDistanceValueValuesEnum(_messages.Enum):
        """The audio type that most closely describes the audio being recognized.

    Values:
      MICROPHONE_DISTANCE_UNSPECIFIED: Audio type is not known.
      NEARFIELD: The audio was captured from a closely placed microphone. Eg.
        phone, dictaphone, or handheld microphone. Generally if there speaker
        is within 1 meter of the microphone.
      MIDFIELD: The speaker if within 3 meters of the microphone.
      FARFIELD: The speaker is more than 3 meters away from the microphone.
    """
        MICROPHONE_DISTANCE_UNSPECIFIED = 0
        NEARFIELD = 1
        MIDFIELD = 2
        FARFIELD = 3

    class OriginalMediaTypeValueValuesEnum(_messages.Enum):
        """The original media the speech was recorded on.

    Values:
      ORIGINAL_MEDIA_TYPE_UNSPECIFIED: Unknown original media type.
      AUDIO: The speech data is an audio recording.
      VIDEO: The speech data originally recorded on a video.
    """
        ORIGINAL_MEDIA_TYPE_UNSPECIFIED = 0
        AUDIO = 1
        VIDEO = 2

    class RecordingDeviceTypeValueValuesEnum(_messages.Enum):
        """The type of device the speech was recorded with.

    Values:
      RECORDING_DEVICE_TYPE_UNSPECIFIED: The recording device is unknown.
      SMARTPHONE: Speech was recorded on a smartphone.
      PC: Speech was recorded using a personal computer or tablet.
      PHONE_LINE: Speech was recorded over a phone line.
      VEHICLE: Speech was recorded in a vehicle.
      OTHER_OUTDOOR_DEVICE: Speech was recorded outdoors.
      OTHER_INDOOR_DEVICE: Speech was recorded indoors.
    """
        RECORDING_DEVICE_TYPE_UNSPECIFIED = 0
        SMARTPHONE = 1
        PC = 2
        PHONE_LINE = 3
        VEHICLE = 4
        OTHER_OUTDOOR_DEVICE = 5
        OTHER_INDOOR_DEVICE = 6
    audioTopic = _messages.StringField(1)
    industryNaicsCodeOfAudio = _messages.IntegerField(2, variant=_messages.Variant.UINT32)
    interactionType = _messages.EnumField('InteractionTypeValueValuesEnum', 3)
    microphoneDistance = _messages.EnumField('MicrophoneDistanceValueValuesEnum', 4)
    originalMediaType = _messages.EnumField('OriginalMediaTypeValueValuesEnum', 5)
    originalMimeType = _messages.StringField(6)
    recordingDeviceName = _messages.StringField(7)
    recordingDeviceType = _messages.EnumField('RecordingDeviceTypeValueValuesEnum', 8)