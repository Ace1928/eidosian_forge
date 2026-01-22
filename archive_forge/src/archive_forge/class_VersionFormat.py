from __future__ import annotations
from typing import MutableMapping, MutableSequence
import proto  # type: ignore
from cloudsdk.google.protobuf import field_mask_pb2  # type: ignore
from cloudsdk.google.protobuf import timestamp_pb2  # type: ignore
class VersionFormat(proto.Enum):
    """Deprecated. This is unused.

        Values:
            VERSION_FORMAT_UNSPECIFIED (0):
                An unspecified format version that will
                default to V2.
            V2 (1):
                ``LogEntry`` version 2 format.
            V1 (2):
                ``LogEntry`` version 1 format.
        """
    VERSION_FORMAT_UNSPECIFIED = 0
    V2 = 1
    V1 = 2