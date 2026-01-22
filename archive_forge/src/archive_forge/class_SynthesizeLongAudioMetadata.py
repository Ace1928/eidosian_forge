from __future__ import annotations
from typing import MutableMapping, MutableSequence
from google.protobuf import timestamp_pb2  # type: ignore
import proto  # type: ignore
from google.cloud.texttospeech_v1.types import cloud_tts
class SynthesizeLongAudioMetadata(proto.Message):
    """Metadata for response returned by the ``SynthesizeLongAudio``
    method.

    Attributes:
        start_time (google.protobuf.timestamp_pb2.Timestamp):
            Time when the request was received.
        last_update_time (google.protobuf.timestamp_pb2.Timestamp):
            Deprecated. Do not use.
        progress_percentage (float):
            The progress of the most recent processing
            update in percentage, ie. 70.0%.
    """
    start_time: timestamp_pb2.Timestamp = proto.Field(proto.MESSAGE, number=1, message=timestamp_pb2.Timestamp)
    last_update_time: timestamp_pb2.Timestamp = proto.Field(proto.MESSAGE, number=2, message=timestamp_pb2.Timestamp)
    progress_percentage: float = proto.Field(proto.DOUBLE, number=3)