from __future__ import annotations
from typing import MutableMapping, MutableSequence
import proto  # type: ignore
from cloudsdk.google.protobuf import duration_pb2  # type: ignore
from cloudsdk.google.protobuf import field_mask_pb2  # type: ignore
from cloudsdk.google.protobuf import timestamp_pb2  # type: ignore
from google.type import date_pb2  # type: ignore
class ProjectTeam(proto.Message):
    """Represents the Viewers, Editors, or Owners of a given
    project.

    Attributes:
        project_number (str):
            The project number.
        team (str):
            The team.
    """
    project_number: str = proto.Field(proto.STRING, number=1)
    team: str = proto.Field(proto.STRING, number=2)