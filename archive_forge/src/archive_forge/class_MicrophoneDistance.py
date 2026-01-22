from __future__ import annotations
from typing import MutableMapping, MutableSequence
from google.protobuf import duration_pb2  # type: ignore
from google.protobuf import timestamp_pb2  # type: ignore
from google.protobuf import wrappers_pb2  # type: ignore
from google.rpc import status_pb2  # type: ignore
import proto  # type: ignore
from google.cloud.speech_v1p1beta1.types import resource
class MicrophoneDistance(proto.Enum):
    """Enumerates the types of capture settings describing an audio
        file.

        Values:
            MICROPHONE_DISTANCE_UNSPECIFIED (0):
                Audio type is not known.
            NEARFIELD (1):
                The audio was captured from a closely placed
                microphone. Eg. phone, dictaphone, or handheld
                microphone. Generally if there speaker is within
                1 meter of the microphone.
            MIDFIELD (2):
                The speaker if within 3 meters of the
                microphone.
            FARFIELD (3):
                The speaker is more than 3 meters away from
                the microphone.
        """
    MICROPHONE_DISTANCE_UNSPECIFIED = 0
    NEARFIELD = 1
    MIDFIELD = 2
    FARFIELD = 3