from __future__ import annotations
import warnings
import typing
from typing import Any, Collection, Optional, Protocol, TypeVar
import google.protobuf.json_format
import google.protobuf.message
import google.protobuf.text_format
import onnx
class _TextProtoSerializer(ProtoSerializer):
    """Serialize and deserialize text proto."""
    supported_format = 'textproto'
    file_extensions = frozenset({'.textproto', '.prototxt', '.pbtxt'})

    def serialize_proto(self, proto: _Proto) -> bytes:
        textproto = google.protobuf.text_format.MessageToString(proto)
        return textproto.encode(_ENCODING)

    def deserialize_proto(self, serialized: bytes | str, proto: _Proto) -> _Proto:
        if not isinstance(serialized, (bytes, str)):
            raise TypeError(f"Parameter 'serialized' must be bytes or str, but got type: {type(serialized)}")
        if isinstance(serialized, bytes):
            serialized = serialized.decode(_ENCODING)
        assert isinstance(serialized, str)
        return google.protobuf.text_format.Parse(serialized, proto)