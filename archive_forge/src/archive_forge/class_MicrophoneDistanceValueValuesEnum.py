from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
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