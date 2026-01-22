from __future__ import annotations
from typing import MutableMapping, MutableSequence
import proto  # type: ignore
from cloudsdk.google.protobuf import duration_pb2  # type: ignore
from cloudsdk.google.protobuf import field_mask_pb2  # type: ignore
from cloudsdk.google.protobuf import timestamp_pb2  # type: ignore
from google.type import date_pb2  # type: ignore
class CustomerEncryption(proto.Message):
    """Describes the Customer-Supplied Encryption Key mechanism used
    to store an Object's data at rest.

    Attributes:
        encryption_algorithm (str):
            The encryption algorithm.
        key_sha256_bytes (bytes):
            SHA256 hash value of the encryption key.
            In raw bytes format (not base64-encoded).
    """
    encryption_algorithm: str = proto.Field(proto.STRING, number=1)
    key_sha256_bytes: bytes = proto.Field(proto.BYTES, number=3)