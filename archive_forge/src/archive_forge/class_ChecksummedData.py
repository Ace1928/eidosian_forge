from __future__ import annotations
from typing import MutableMapping, MutableSequence
import proto  # type: ignore
from cloudsdk.google.protobuf import duration_pb2  # type: ignore
from cloudsdk.google.protobuf import field_mask_pb2  # type: ignore
from cloudsdk.google.protobuf import timestamp_pb2  # type: ignore
from google.type import date_pb2  # type: ignore
class ChecksummedData(proto.Message):
    """Message used to convey content being read or written, along
    with an optional checksum.


    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        content (bytes):
            Optional. The data.
        crc32c (int):
            If set, the CRC32C digest of the content
            field.

            This field is a member of `oneof`_ ``_crc32c``.
    """
    content: bytes = proto.Field(proto.BYTES, number=1)
    crc32c: int = proto.Field(proto.FIXED32, number=2, optional=True)