from __future__ import annotations
from typing import MutableMapping, MutableSequence
import proto  # type: ignore
from google.cloud.pubsublite_v1.types import common
from cloudsdk.google.protobuf import field_mask_pb2  # type: ignore
from cloudsdk.google.protobuf import timestamp_pb2  # type: ignore
A named position with respect to the message backlog.

        Values:
            NAMED_TARGET_UNSPECIFIED (0):
                Unspecified named target. Do not use.
            TAIL (1):
                Seek to the oldest retained message.
            HEAD (2):
                Seek past all recently published messages,
                skipping the entire message backlog.
        