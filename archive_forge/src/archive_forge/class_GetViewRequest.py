from __future__ import annotations
from typing import MutableMapping, MutableSequence
import proto  # type: ignore
from cloudsdk.google.protobuf import field_mask_pb2  # type: ignore
from cloudsdk.google.protobuf import timestamp_pb2  # type: ignore
class GetViewRequest(proto.Message):
    """The parameters to ``GetView``.

    Attributes:
        name (str):
            Required. The resource name of the policy:

            ::

                "projects/[PROJECT_ID]/locations/[LOCATION_ID]/buckets/[BUCKET_ID]/views/[VIEW_ID]"

            For example:

            ``"projects/my-project/locations/global/buckets/my-bucket/views/my-view"``
    """
    name: str = proto.Field(proto.STRING, number=1)