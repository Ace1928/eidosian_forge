from __future__ import annotations
from typing import MutableMapping, MutableSequence
import proto  # type: ignore
from google.cloud.pubsublite_v1.types import common
from cloudsdk.google.protobuf import field_mask_pb2  # type: ignore
from cloudsdk.google.protobuf import timestamp_pb2  # type: ignore
class CreateTopicRequest(proto.Message):
    """Request for CreateTopic.

    Attributes:
        parent (str):
            Required. The parent location in which to create the topic.
            Structured like
            ``projects/{project_number}/locations/{location}``.
        topic (google.cloud.pubsublite_v1.types.Topic):
            Required. Configuration of the topic to create. Its ``name``
            field is ignored.
        topic_id (str):
            Required. The ID to use for the topic, which will become the
            final component of the topic's name.

            This value is structured like: ``my-topic-name``.
    """
    parent: str = proto.Field(proto.STRING, number=1)
    topic: common.Topic = proto.Field(proto.MESSAGE, number=2, message=common.Topic)
    topic_id: str = proto.Field(proto.STRING, number=3)