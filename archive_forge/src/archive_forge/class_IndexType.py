from __future__ import annotations
from typing import MutableMapping, MutableSequence
import proto  # type: ignore
from cloudsdk.google.protobuf import field_mask_pb2  # type: ignore
from cloudsdk.google.protobuf import timestamp_pb2  # type: ignore
class IndexType(proto.Enum):
    """IndexType is used for custom indexing. It describes the type
    of an indexed field.

    Values:
        INDEX_TYPE_UNSPECIFIED (0):
            The index's type is unspecified.
        INDEX_TYPE_STRING (1):
            The index is a string-type index.
        INDEX_TYPE_INTEGER (2):
            The index is a integer-type index.
    """
    INDEX_TYPE_UNSPECIFIED = 0
    INDEX_TYPE_STRING = 1
    INDEX_TYPE_INTEGER = 2