from __future__ import annotations
from typing import MutableMapping, MutableSequence
import proto  # type: ignore
from cloudsdk.google.protobuf import timestamp_pb2  # type: ignore
class CreateSchemaRequest(proto.Message):
    """Request for the CreateSchema method.

    Attributes:
        parent (str):
            Required. The name of the project in which to create the
            schema. Format is ``projects/{project-id}``.
        schema (google.pubsub_v1.types.Schema):
            Required. The schema object to create.

            This schema's ``name`` parameter is ignored. The schema
            object returned by CreateSchema will have a ``name`` made
            using the given ``parent`` and ``schema_id``.
        schema_id (str):
            The ID to use for the schema, which will become the final
            component of the schema's resource name.

            See
            https://cloud.google.com/pubsub/docs/admin#resource_names
            for resource name constraints.
    """
    parent: str = proto.Field(proto.STRING, number=1)
    schema: 'Schema' = proto.Field(proto.MESSAGE, number=2, message='Schema')
    schema_id: str = proto.Field(proto.STRING, number=3)