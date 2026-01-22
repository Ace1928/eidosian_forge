from __future__ import annotations
from typing import MutableMapping, MutableSequence
import proto  # type: ignore
from cloudsdk.google.protobuf import duration_pb2  # type: ignore
from cloudsdk.google.protobuf import field_mask_pb2  # type: ignore
from cloudsdk.google.protobuf import timestamp_pb2  # type: ignore
from google.type import date_pb2  # type: ignore
class GetServiceAccountRequest(proto.Message):
    """Request message for GetServiceAccount.

    Attributes:
        project (str):
            Required. Project ID, in the format of
            "projects/{projectIdentifier}".
            {projectIdentifier} can be the project ID or
            project number.
    """
    project: str = proto.Field(proto.STRING, number=1)