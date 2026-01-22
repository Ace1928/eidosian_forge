from __future__ import annotations
from typing import MutableMapping, MutableSequence
import proto  # type: ignore
from cloudsdk.google.protobuf import timestamp_pb2  # type: ignore
class SchemaView(proto.Enum):
    """View of Schema object fields to be returned by GetSchema and
    ListSchemas.

    Values:
        SCHEMA_VIEW_UNSPECIFIED (0):
            The default / unset value.
            The API will default to the BASIC view.
        BASIC (1):
            Include the name and type of the schema, but
            not the definition.
        FULL (2):
            Include all Schema object fields.
    """
    SCHEMA_VIEW_UNSPECIFIED = 0
    BASIC = 1
    FULL = 2