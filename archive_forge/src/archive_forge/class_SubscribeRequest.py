from __future__ import annotations
from typing import MutableMapping, MutableSequence
import proto  # type: ignore
from google.cloud.pubsublite_v1.types import common
class SubscribeRequest(proto.Message):
    """A request sent from the client to the server on a stream.

    This message has `oneof`_ fields (mutually exclusive fields).
    For each oneof, at most one member field can be set at the same time.
    Setting any member of the oneof automatically clears all other
    members.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        initial (google.cloud.pubsublite_v1.types.InitialSubscribeRequest):
            Initial request on the stream.

            This field is a member of `oneof`_ ``request``.
        seek (google.cloud.pubsublite_v1.types.SeekRequest):
            Request to update the stream's delivery
            cursor.

            This field is a member of `oneof`_ ``request``.
        flow_control (google.cloud.pubsublite_v1.types.FlowControlRequest):
            Request to grant tokens to the server,

            This field is a member of `oneof`_ ``request``.
    """
    initial: 'InitialSubscribeRequest' = proto.Field(proto.MESSAGE, number=1, oneof='request', message='InitialSubscribeRequest')
    seek: 'SeekRequest' = proto.Field(proto.MESSAGE, number=2, oneof='request', message='SeekRequest')
    flow_control: 'FlowControlRequest' = proto.Field(proto.MESSAGE, number=3, oneof='request', message='FlowControlRequest')