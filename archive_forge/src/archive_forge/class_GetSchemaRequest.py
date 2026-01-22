from __future__ import annotations
from typing import MutableMapping, MutableSequence
import proto  # type: ignore
from cloudsdk.google.protobuf import timestamp_pb2  # type: ignore
class GetSchemaRequest(proto.Message):
    """Request for the GetSchema method.

    Attributes:
        name (str):
            Required. The name of the schema to get. Format is
            ``projects/{project}/schemas/{schema}``.
        view (google.pubsub_v1.types.SchemaView):
            The set of fields to return in the response. If not set,
            returns a Schema with all fields filled out. Set to
            ``BASIC`` to omit the ``definition``.
    """
    name: str = proto.Field(proto.STRING, number=1)
    view: 'SchemaView' = proto.Field(proto.ENUM, number=2, enum='SchemaView')