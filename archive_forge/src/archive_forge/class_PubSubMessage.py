from __future__ import annotations
from typing import MutableMapping, MutableSequence
import proto  # type: ignore
from cloudsdk.google.protobuf import duration_pb2  # type: ignore
from cloudsdk.google.protobuf import timestamp_pb2  # type: ignore
class PubSubMessage(proto.Message):
    """A message that is published by publishers and delivered to
    subscribers.

    Attributes:
        key (bytes):
            The key used for routing messages to
            partitions or for compaction (e.g., keep the
            last N messages per key). If the key is empty,
            the message is routed to an arbitrary partition.
        data (bytes):
            The payload of the message.
        attributes (MutableMapping[str, google.cloud.pubsublite_v1.types.AttributeValues]):
            Optional attributes that can be used for
            message metadata/headers.
        event_time (google.protobuf.timestamp_pb2.Timestamp):
            An optional, user-specified event time.
    """
    key: bytes = proto.Field(proto.BYTES, number=1)
    data: bytes = proto.Field(proto.BYTES, number=2)
    attributes: MutableMapping[str, 'AttributeValues'] = proto.MapField(proto.STRING, proto.MESSAGE, number=3, message='AttributeValues')
    event_time: timestamp_pb2.Timestamp = proto.Field(proto.MESSAGE, number=4, message=timestamp_pb2.Timestamp)