from __future__ import annotations
from typing import MutableMapping, MutableSequence
from google.protobuf import duration_pb2  # type: ignore
from google.protobuf import timestamp_pb2  # type: ignore
from google.protobuf import wrappers_pb2  # type: ignore
from google.rpc import status_pb2  # type: ignore
import proto  # type: ignore
from google.cloud.speech_v1p1beta1.types import resource
class RecordingDeviceType(proto.Enum):
    """The type of device the speech was recorded with.

        Values:
            RECORDING_DEVICE_TYPE_UNSPECIFIED (0):
                The recording device is unknown.
            SMARTPHONE (1):
                Speech was recorded on a smartphone.
            PC (2):
                Speech was recorded using a personal computer
                or tablet.
            PHONE_LINE (3):
                Speech was recorded over a phone line.
            VEHICLE (4):
                Speech was recorded in a vehicle.
            OTHER_OUTDOOR_DEVICE (5):
                Speech was recorded outdoors.
            OTHER_INDOOR_DEVICE (6):
                Speech was recorded indoors.
        """
    RECORDING_DEVICE_TYPE_UNSPECIFIED = 0
    SMARTPHONE = 1
    PC = 2
    PHONE_LINE = 3
    VEHICLE = 4
    OTHER_OUTDOOR_DEVICE = 5
    OTHER_INDOOR_DEVICE = 6