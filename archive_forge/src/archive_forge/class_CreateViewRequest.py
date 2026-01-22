from __future__ import annotations
from typing import MutableMapping, MutableSequence
import proto  # type: ignore
from cloudsdk.google.protobuf import field_mask_pb2  # type: ignore
from cloudsdk.google.protobuf import timestamp_pb2  # type: ignore
class CreateViewRequest(proto.Message):
    """The parameters to ``CreateView``.

    Attributes:
        parent (str):
            Required. The bucket in which to create the view

            ::

                `"projects/[PROJECT_ID]/locations/[LOCATION_ID]/buckets/[BUCKET_ID]"`

            For example:

            ``"projects/my-project/locations/global/buckets/my-bucket"``
        view_id (str):
            Required. A client-assigned identifier such as
            ``"my-view"``. Identifiers are limited to 100 characters and
            can include only letters, digits, underscores, hyphens, and
            periods.
        view (googlecloudsdk.generated_clients.gapic_clients.logging_v2.types.LogView):
            Required. The new view.
    """
    parent: str = proto.Field(proto.STRING, number=1)
    view_id: str = proto.Field(proto.STRING, number=2)
    view: 'LogView' = proto.Field(proto.MESSAGE, number=3, message='LogView')