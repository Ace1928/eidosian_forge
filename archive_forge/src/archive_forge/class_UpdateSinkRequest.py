from __future__ import annotations
from typing import MutableMapping, MutableSequence
import proto  # type: ignore
from cloudsdk.google.protobuf import field_mask_pb2  # type: ignore
from cloudsdk.google.protobuf import timestamp_pb2  # type: ignore
class UpdateSinkRequest(proto.Message):
    """The parameters to ``UpdateSink``.

    Attributes:
        sink_name (str):
            Required. The full resource name of the sink to update,
            including the parent resource and the sink identifier:

            ::

                "projects/[PROJECT_ID]/sinks/[SINK_ID]"
                "organizations/[ORGANIZATION_ID]/sinks/[SINK_ID]"
                "billingAccounts/[BILLING_ACCOUNT_ID]/sinks/[SINK_ID]"
                "folders/[FOLDER_ID]/sinks/[SINK_ID]"

            For example:

            ``"projects/my-project/sinks/my-sink"``
        sink (googlecloudsdk.generated_clients.gapic_clients.logging_v2.types.LogSink):
            Required. The updated sink, whose name is the same
            identifier that appears as part of ``sink_name``.
        unique_writer_identity (bool):
            Optional. See
            [sinks.create][google.logging.v2.ConfigServiceV2.CreateSink]
            for a description of this field. When updating a sink, the
            effect of this field on the value of ``writer_identity`` in
            the updated sink depends on both the old and new values of
            this field:

            -  If the old and new values of this field are both false or
               both true, then there is no change to the sink's
               ``writer_identity``.
            -  If the old value is false and the new value is true, then
               ``writer_identity`` is changed to a `service
               agent <https://cloud.google.com/iam/docs/service-account-types#service-agents>`__
               owned by Cloud Logging.
            -  It is an error if the old value is true and the new value
               is set to false or defaulted to false.
        custom_writer_identity (str):
            Optional. A service account provided by the caller that will
            be used to write the log entries. The format must be
            ``serviceAccount:some@email``. This field can only be
            specified if you are routing logs to a destination outside
            this sink's project. If not specified, a Logging service
            account will automatically be generated.
        update_mask (google.protobuf.field_mask_pb2.FieldMask):
            Optional. Field mask that specifies the fields in ``sink``
            that need an update. A sink field will be overwritten if,
            and only if, it is in the update mask. ``name`` and output
            only fields cannot be updated.

            An empty ``updateMask`` is temporarily treated as using the
            following mask for backwards compatibility purposes:

            ``destination,filter,includeChildren``

            At some point in the future, behavior will be removed and
            specifying an empty ``updateMask`` will be an error.

            For a detailed ``FieldMask`` definition, see
            https://developers.google.com/protocol-buffers/docs/reference/google.protobuf#google.protobuf.FieldMask

            For example: ``updateMask=filter``
    """
    sink_name: str = proto.Field(proto.STRING, number=1)
    sink: 'LogSink' = proto.Field(proto.MESSAGE, number=2, message='LogSink')
    unique_writer_identity: bool = proto.Field(proto.BOOL, number=3)
    custom_writer_identity: str = proto.Field(proto.STRING, number=5)
    update_mask: field_mask_pb2.FieldMask = proto.Field(proto.MESSAGE, number=4, message=field_mask_pb2.FieldMask)