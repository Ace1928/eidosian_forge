from __future__ import annotations
from typing import MutableMapping, MutableSequence
from google.protobuf import duration_pb2  # type: ignore
from google.protobuf import field_mask_pb2  # type: ignore
from google.protobuf import timestamp_pb2  # type: ignore
from google.rpc import status_pb2  # type: ignore
import proto  # type: ignore
class MultiChannelMode(proto.Enum):
    """Options for how to recognize multi-channel audio.

        Values:
            MULTI_CHANNEL_MODE_UNSPECIFIED (0):
                Default value for the multi-channel mode. If
                the audio contains multiple channels, only the
                first channel will be transcribed; other
                channels will be ignored.
            SEPARATE_RECOGNITION_PER_CHANNEL (1):
                If selected, each channel in the provided audio is
                transcribed independently. This cannot be selected if the
                selected [model][google.cloud.speech.v2.Recognizer.model] is
                ``latest_short``.
        """
    MULTI_CHANNEL_MODE_UNSPECIFIED = 0
    SEPARATE_RECOGNITION_PER_CHANNEL = 1