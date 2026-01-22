from __future__ import annotations
from typing import MutableMapping, MutableSequence
import proto  # type: ignore
from cloudsdk.google.protobuf import duration_pb2  # type: ignore
from cloudsdk.google.protobuf import field_mask_pb2  # type: ignore
from cloudsdk.google.protobuf import timestamp_pb2  # type: ignore
from google.type import date_pb2  # type: ignore
class UpdateHmacKeyRequest(proto.Message):
    """Request object to update an HMAC key state.
    HmacKeyMetadata.state is required and the only writable field in
    UpdateHmacKey operation. Specifying fields other than state will
    result in an error.

    Attributes:
        hmac_key (googlecloudsdk.generated_clients.gapic_clients.storage_v2.types.HmacKeyMetadata):
            Required. The HMAC key to update. If present, the hmac_key's
            ``id`` field will be used to identify the key. Otherwise,
            the hmac_key's access_id and project fields will be used to
            identify the key.
        update_mask (google.protobuf.field_mask_pb2.FieldMask):
            Update mask for hmac_key. Not specifying any fields will
            mean only the ``state`` field is updated to the value
            specified in ``hmac_key``.
    """
    hmac_key: 'HmacKeyMetadata' = proto.Field(proto.MESSAGE, number=1, message='HmacKeyMetadata')
    update_mask: field_mask_pb2.FieldMask = proto.Field(proto.MESSAGE, number=3, message=field_mask_pb2.FieldMask)