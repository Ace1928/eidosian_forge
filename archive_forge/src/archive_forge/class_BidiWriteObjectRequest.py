from __future__ import annotations
from typing import MutableMapping, MutableSequence
import proto  # type: ignore
from cloudsdk.google.protobuf import duration_pb2  # type: ignore
from cloudsdk.google.protobuf import field_mask_pb2  # type: ignore
from cloudsdk.google.protobuf import timestamp_pb2  # type: ignore
from google.type import date_pb2  # type: ignore
class BidiWriteObjectRequest(proto.Message):
    """Request message for BidiWriteObject.

    This message has `oneof`_ fields (mutually exclusive fields).
    For each oneof, at most one member field can be set at the same time.
    Setting any member of the oneof automatically clears all other
    members.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        upload_id (str):
            For resumable uploads. This should be the ``upload_id``
            returned from a call to ``StartResumableWriteResponse``.

            This field is a member of `oneof`_ ``first_message``.
        write_object_spec (googlecloudsdk.generated_clients.gapic_clients.storage_v2.types.WriteObjectSpec):
            For non-resumable uploads. Describes the
            overall upload, including the destination bucket
            and object name, preconditions, etc.

            This field is a member of `oneof`_ ``first_message``.
        write_offset (int):
            Required. The offset from the beginning of the object at
            which the data should be written.

            In the first ``WriteObjectRequest`` of a ``WriteObject()``
            action, it indicates the initial offset for the ``Write()``
            call. The value **must** be equal to the ``persisted_size``
            that a call to ``QueryWriteStatus()`` would return (0 if
            this is the first write to the object).

            On subsequent calls, this value **must** be no larger than
            the sum of the first ``write_offset`` and the sizes of all
            ``data`` chunks sent previously on this stream.

            An invalid value will cause an error.
        checksummed_data (googlecloudsdk.generated_clients.gapic_clients.storage_v2.types.ChecksummedData):
            The data to insert. If a crc32c checksum is
            provided that doesn't match the checksum
            computed by the service, the request will fail.

            This field is a member of `oneof`_ ``data``.
        object_checksums (googlecloudsdk.generated_clients.gapic_clients.storage_v2.types.ObjectChecksums):
            Checksums for the complete object. If the checksums computed
            by the service don't match the specified checksums the call
            will fail. May only be provided in the first or last request
            (either with first_message, or finish_write set).
        state_lookup (bool):
            For each BidiWriteObjectRequest where state_lookup is
            ``true`` or the client closes the stream, the service will
            send a BidiWriteObjectResponse containing the current
            persisted size. The persisted size sent in responses covers
            all the bytes the server has persisted thus far and can be
            used to decide what data is safe for the client to drop.
            Note that the object's current size reported by the
            BidiWriteObjectResponse may lag behind the number of bytes
            written by the client. This field is ignored if
            ``finish_write`` is set to true.
        flush (bool):
            Persists data written on the stream, up to and including the
            current message, to permanent storage. This option should be
            used sparingly as it may reduce performance. Ongoing writes
            will periodically be persisted on the server even when
            ``flush`` is not set. This field is ignored if
            ``finish_write`` is set to true since there's no need to
            checkpoint or flush if this message completes the write.
        finish_write (bool):
            If ``true``, this indicates that the write is complete.
            Sending any ``WriteObjectRequest``\\ s subsequent to one in
            which ``finish_write`` is ``true`` will cause an error. For
            a non-resumable write (where the upload_id was not set in
            the first message), it is an error not to set this field in
            the final message of the stream.
        common_object_request_params (googlecloudsdk.generated_clients.gapic_clients.storage_v2.types.CommonObjectRequestParams):
            A set of parameters common to Storage API
            requests concerning an object.
    """
    upload_id: str = proto.Field(proto.STRING, number=1, oneof='first_message')
    write_object_spec: 'WriteObjectSpec' = proto.Field(proto.MESSAGE, number=2, oneof='first_message', message='WriteObjectSpec')
    write_offset: int = proto.Field(proto.INT64, number=3)
    checksummed_data: 'ChecksummedData' = proto.Field(proto.MESSAGE, number=4, oneof='data', message='ChecksummedData')
    object_checksums: 'ObjectChecksums' = proto.Field(proto.MESSAGE, number=6, message='ObjectChecksums')
    state_lookup: bool = proto.Field(proto.BOOL, number=7)
    flush: bool = proto.Field(proto.BOOL, number=8)
    finish_write: bool = proto.Field(proto.BOOL, number=9)
    common_object_request_params: 'CommonObjectRequestParams' = proto.Field(proto.MESSAGE, number=10, message='CommonObjectRequestParams')