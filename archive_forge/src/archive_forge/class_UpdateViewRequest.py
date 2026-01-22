from __future__ import annotations
from typing import MutableMapping, MutableSequence
import proto  # type: ignore
from cloudsdk.google.protobuf import field_mask_pb2  # type: ignore
from cloudsdk.google.protobuf import timestamp_pb2  # type: ignore
class UpdateViewRequest(proto.Message):
    """The parameters to ``UpdateView``.

    Attributes:
        name (str):
            Required. The full resource name of the view to update

            ::

                "projects/[PROJECT_ID]/locations/[LOCATION_ID]/buckets/[BUCKET_ID]/views/[VIEW_ID]"

            For example:

            ``"projects/my-project/locations/global/buckets/my-bucket/views/my-view"``
        view (googlecloudsdk.generated_clients.gapic_clients.logging_v2.types.LogView):
            Required. The updated view.
        update_mask (google.protobuf.field_mask_pb2.FieldMask):
            Optional. Field mask that specifies the fields in ``view``
            that need an update. A field will be overwritten if, and
            only if, it is in the update mask. ``name`` and output only
            fields cannot be updated.

            For a detailed ``FieldMask`` definition, see
            https://developers.google.com/protocol-buffers/docs/reference/google.protobuf#google.protobuf.FieldMask

            For example: ``updateMask=filter``
    """
    name: str = proto.Field(proto.STRING, number=1)
    view: 'LogView' = proto.Field(proto.MESSAGE, number=2, message='LogView')
    update_mask: field_mask_pb2.FieldMask = proto.Field(proto.MESSAGE, number=4, message=field_mask_pb2.FieldMask)