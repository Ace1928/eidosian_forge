from __future__ import annotations
from typing import MutableMapping, MutableSequence
import proto  # type: ignore
from google.cloud.pubsublite_v1.types import common
class InitialPartitionAssignmentRequest(proto.Message):
    """The first request that must be sent on a newly-opened stream.
    The client must wait for the response before sending subsequent
    requests on the stream.

    Attributes:
        subscription (str):
            The subscription name. Structured like:
            projects/<project number>/locations/<zone
            name>/subscriptions/<subscription id>
        client_id (bytes):
            An opaque, unique client identifier. This
            field must be exactly 16 bytes long and is
            interpreted as an unsigned 128 bit integer.
            Other size values will be rejected and the
            stream will be failed with a non-retryable
            error.
            This field is large enough to fit a uuid from
            standard uuid algorithms like uuid1 or uuid4,
            which should be used to generate this number.
            The same identifier should be reused following
            disconnections with retryable stream errors.
    """
    subscription: str = proto.Field(proto.STRING, number=1)
    client_id: bytes = proto.Field(proto.BYTES, number=2)