from __future__ import annotations
from typing import MutableMapping, MutableSequence
import proto  # type: ignore
from google.cloud.pubsublite_v1.types import common
class SubscribeResponse(proto.Message):
    """Response to SubscribeRequest.

    This message has `oneof`_ fields (mutually exclusive fields).
    For each oneof, at most one member field can be set at the same time.
    Setting any member of the oneof automatically clears all other
    members.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        initial (google.cloud.pubsublite_v1.types.InitialSubscribeResponse):
            Initial response on the stream.

            This field is a member of `oneof`_ ``response``.
        seek (google.cloud.pubsublite_v1.types.SeekResponse):
            Response to a Seek operation.

            This field is a member of `oneof`_ ``response``.
        messages (google.cloud.pubsublite_v1.types.MessageResponse):
            Response containing messages from the topic
            partition.

            This field is a member of `oneof`_ ``response``.
    """
    initial: 'InitialSubscribeResponse' = proto.Field(proto.MESSAGE, number=1, oneof='response', message='InitialSubscribeResponse')
    seek: 'SeekResponse' = proto.Field(proto.MESSAGE, number=2, oneof='response', message='SeekResponse')
    messages: 'MessageResponse' = proto.Field(proto.MESSAGE, number=3, oneof='response', message='MessageResponse')