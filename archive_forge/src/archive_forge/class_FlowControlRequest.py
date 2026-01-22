from __future__ import annotations
from typing import MutableMapping, MutableSequence
import proto  # type: ignore
from google.cloud.pubsublite_v1.types import common
class FlowControlRequest(proto.Message):
    """Request to grant tokens to the server, requesting delivery of
    messages when they become available.

    Attributes:
        allowed_messages (int):
            The number of message tokens to grant. Must
            be greater than or equal to 0.
        allowed_bytes (int):
            The number of byte tokens to grant. Must be
            greater than or equal to 0.
    """
    allowed_messages: int = proto.Field(proto.INT64, number=1)
    allowed_bytes: int = proto.Field(proto.INT64, number=2)