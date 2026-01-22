from __future__ import annotations
from typing import MutableMapping, MutableSequence
import proto  # type: ignore
from cloudsdk.google.protobuf import duration_pb2  # type: ignore
from cloudsdk.google.protobuf import field_mask_pb2  # type: ignore
from cloudsdk.google.protobuf import timestamp_pb2  # type: ignore
from google.type import date_pb2  # type: ignore
class WriteObjectResponse(proto.Message):
    """Response message for WriteObject.

    This message has `oneof`_ fields (mutually exclusive fields).
    For each oneof, at most one member field can be set at the same time.
    Setting any member of the oneof automatically clears all other
    members.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        persisted_size (int):
            The total number of bytes that have been processed for the
            given object from all ``WriteObject`` calls. Only set if the
            upload has not finalized.

            This field is a member of `oneof`_ ``write_status``.
        resource (googlecloudsdk.generated_clients.gapic_clients.storage_v2.types.Object):
            A resource containing the metadata for the
            uploaded object. Only set if the upload has
            finalized.

            This field is a member of `oneof`_ ``write_status``.
    """
    persisted_size: int = proto.Field(proto.INT64, number=1, oneof='write_status')
    resource: 'Object' = proto.Field(proto.MESSAGE, number=2, oneof='write_status', message='Object')