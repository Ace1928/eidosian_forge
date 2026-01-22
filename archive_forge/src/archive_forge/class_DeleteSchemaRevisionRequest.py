from __future__ import annotations
from typing import MutableMapping, MutableSequence
import proto  # type: ignore
from cloudsdk.google.protobuf import timestamp_pb2  # type: ignore
class DeleteSchemaRevisionRequest(proto.Message):
    """Request for the ``DeleteSchemaRevision`` method.

    Attributes:
        name (str):
            Required. The name of the schema revision to be deleted,
            with a revision ID explicitly included.

            Example: ``projects/123/schemas/my-schema@c7cfa2a8``
        revision_id (str):
            Optional. This field is deprecated and should not be used
            for specifying the revision ID. The revision ID should be
            specified via the ``name`` parameter.
    """
    name: str = proto.Field(proto.STRING, number=1)
    revision_id: str = proto.Field(proto.STRING, number=2)