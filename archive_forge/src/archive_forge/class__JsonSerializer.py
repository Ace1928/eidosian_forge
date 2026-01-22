from __future__ import annotations
import warnings
import typing
from typing import Any, Collection, Optional, Protocol, TypeVar
import google.protobuf.json_format
import google.protobuf.message
import google.protobuf.text_format
import onnx
class _JsonSerializer(ProtoSerializer):
    """Serialize and deserialize JSON."""
    supported_format = 'json'
    file_extensions = frozenset({'.json', '.onnxjson'})

    def serialize_proto(self, proto: _Proto) -> bytes:
        json_message = google.protobuf.json_format.MessageToJson(proto, preserving_proto_field_name=True)
        return json_message.encode(_ENCODING)

    def deserialize_proto(self, serialized: bytes | str, proto: _Proto) -> _Proto:
        if not isinstance(serialized, (bytes, str)):
            raise TypeError(f"Parameter 'serialized' must be bytes or str, but got type: {type(serialized)}")
        if isinstance(serialized, bytes):
            serialized = serialized.decode(_ENCODING)
        assert isinstance(serialized, str)
        return google.protobuf.json_format.Parse(serialized, proto)