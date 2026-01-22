from __future__ import annotations
from typing import MutableMapping, MutableSequence
import proto  # type: ignore
from cloudsdk.google.protobuf import duration_pb2  # type: ignore
from cloudsdk.google.protobuf import field_mask_pb2  # type: ignore
from cloudsdk.google.protobuf import timestamp_pb2  # type: ignore
from google.type import date_pb2  # type: ignore
class CreateHmacKeyResponse(proto.Message):
    """Create hmac response.  The only time the secret for an HMAC
    will be returned.

    Attributes:
        metadata (googlecloudsdk.generated_clients.gapic_clients.storage_v2.types.HmacKeyMetadata):
            Key metadata.
        secret_key_bytes (bytes):
            HMAC key secret material.
            In raw bytes format (not base64-encoded).
    """
    metadata: 'HmacKeyMetadata' = proto.Field(proto.MESSAGE, number=1, message='HmacKeyMetadata')
    secret_key_bytes: bytes = proto.Field(proto.BYTES, number=3)