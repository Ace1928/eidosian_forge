from __future__ import annotations
import warnings
import typing
from typing import Any, Collection, Optional, Protocol, TypeVar
import google.protobuf.json_format
import google.protobuf.message
import google.protobuf.text_format
import onnx
class _ProtobufSerializer(ProtoSerializer):
    """Serialize and deserialize protobuf message."""
    supported_format = 'protobuf'
    file_extensions = frozenset({'.onnx', '.pb'})

    def serialize_proto(self, proto: _Proto) -> bytes:
        if hasattr(proto, 'SerializeToString') and callable(proto.SerializeToString):
            try:
                result = proto.SerializeToString()
            except ValueError as e:
                if proto.ByteSize() >= onnx.checker.MAXIMUM_PROTOBUF:
                    raise ValueError('The proto size is larger than the 2 GB limit. Please use save_as_external_data to save tensors separately from the model file.') from e
                raise
            return result
        raise TypeError(f'No SerializeToString method is detected.\ntype is {type(proto)}')

    def deserialize_proto(self, serialized: bytes, proto: _Proto) -> _Proto:
        if not isinstance(serialized, bytes):
            raise TypeError(f"Parameter 'serialized' must be bytes, but got type: {type(serialized)}")
        decoded = typing.cast(Optional[int], proto.ParseFromString(serialized))
        if decoded is not None and decoded != len(serialized):
            raise google.protobuf.message.DecodeError(f'Protobuf decoding consumed too few bytes: {decoded} out of {len(serialized)}')
        return proto