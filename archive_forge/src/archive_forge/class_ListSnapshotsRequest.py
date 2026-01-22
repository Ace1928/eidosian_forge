from __future__ import annotations
from typing import MutableMapping, MutableSequence
import proto  # type: ignore
from cloudsdk.google.protobuf import duration_pb2  # type: ignore
from cloudsdk.google.protobuf import field_mask_pb2  # type: ignore
from cloudsdk.google.protobuf import timestamp_pb2  # type: ignore
from google.pubsub_v1.types import schema as gp_schema
class ListSnapshotsRequest(proto.Message):
    """Request for the ``ListSnapshots`` method.

    Attributes:
        project (str):
            Required. The name of the project in which to list
            snapshots. Format is ``projects/{project-id}``.
        page_size (int):
            Maximum number of snapshots to return.
        page_token (str):
            The value returned by the last ``ListSnapshotsResponse``;
            indicates that this is a continuation of a prior
            ``ListSnapshots`` call, and that the system should return
            the next page of data.
    """
    project: str = proto.Field(proto.STRING, number=1)
    page_size: int = proto.Field(proto.INT32, number=2)
    page_token: str = proto.Field(proto.STRING, number=3)