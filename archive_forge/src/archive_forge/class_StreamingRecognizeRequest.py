from __future__ import annotations
from typing import MutableMapping, MutableSequence
from google.protobuf import duration_pb2  # type: ignore
from google.protobuf import timestamp_pb2  # type: ignore
from google.protobuf import wrappers_pb2  # type: ignore
from google.rpc import status_pb2  # type: ignore
import proto  # type: ignore
from google.cloud.speech_v1p1beta1.types import resource
class StreamingRecognizeRequest(proto.Message):
    """The top-level message sent by the client for the
    ``StreamingRecognize`` method. Multiple
    ``StreamingRecognizeRequest`` messages are sent. The first message
    must contain a ``streaming_config`` message and must not contain
    ``audio_content``. All subsequent messages must contain
    ``audio_content`` and must not contain a ``streaming_config``
    message.

    This message has `oneof`_ fields (mutually exclusive fields).
    For each oneof, at most one member field can be set at the same time.
    Setting any member of the oneof automatically clears all other
    members.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        streaming_config (google.cloud.speech_v1p1beta1.types.StreamingRecognitionConfig):
            Provides information to the recognizer that specifies how to
            process the request. The first ``StreamingRecognizeRequest``
            message must contain a ``streaming_config`` message.

            This field is a member of `oneof`_ ``streaming_request``.
        audio_content (bytes):
            The audio data to be recognized. Sequential chunks of audio
            data are sent in sequential ``StreamingRecognizeRequest``
            messages. The first ``StreamingRecognizeRequest`` message
            must not contain ``audio_content`` data and all subsequent
            ``StreamingRecognizeRequest`` messages must contain
            ``audio_content`` data. The audio bytes must be encoded as
            specified in ``RecognitionConfig``. Note: as with all bytes
            fields, proto buffers use a pure binary representation (not
            base64). See `content
            limits <https://cloud.google.com/speech-to-text/quotas#content>`__.

            This field is a member of `oneof`_ ``streaming_request``.
    """
    streaming_config: 'StreamingRecognitionConfig' = proto.Field(proto.MESSAGE, number=1, oneof='streaming_request', message='StreamingRecognitionConfig')
    audio_content: bytes = proto.Field(proto.BYTES, number=2, oneof='streaming_request')