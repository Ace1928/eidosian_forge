from __future__ import annotations
from typing import MutableMapping, MutableSequence
import proto  # type: ignore
from google.cloud.pubsublite_v1.types import common
class NamedTarget(proto.Enum):
    """A special target in the partition that takes no other
        parameters.

        Values:
            NAMED_TARGET_UNSPECIFIED (0):
                Default value. This value is unused.
            HEAD (1):
                A target corresponding to the most recently
                published message in the partition.
            COMMITTED_CURSOR (2):
                A target corresponding to the committed
                cursor for the given subscription and topic
                partition.
        """
    NAMED_TARGET_UNSPECIFIED = 0
    HEAD = 1
    COMMITTED_CURSOR = 2