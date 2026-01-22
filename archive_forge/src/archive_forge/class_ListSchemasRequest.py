from __future__ import annotations
from typing import MutableMapping, MutableSequence
import proto  # type: ignore
from cloudsdk.google.protobuf import timestamp_pb2  # type: ignore
class ListSchemasRequest(proto.Message):
    """Request for the ``ListSchemas`` method.

    Attributes:
        parent (str):
            Required. The name of the project in which to list schemas.
            Format is ``projects/{project-id}``.
        view (google.pubsub_v1.types.SchemaView):
            The set of Schema fields to return in the response. If not
            set, returns Schemas with ``name`` and ``type``, but not
            ``definition``. Set to ``FULL`` to retrieve all fields.
        page_size (int):
            Maximum number of schemas to return.
        page_token (str):
            The value returned by the last ``ListSchemasResponse``;
            indicates that this is a continuation of a prior
            ``ListSchemas`` call, and that the system should return the
            next page of data.
    """
    parent: str = proto.Field(proto.STRING, number=1)
    view: 'SchemaView' = proto.Field(proto.ENUM, number=2, enum='SchemaView')
    page_size: int = proto.Field(proto.INT32, number=3)
    page_token: str = proto.Field(proto.STRING, number=4)