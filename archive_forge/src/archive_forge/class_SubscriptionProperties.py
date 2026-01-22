from __future__ import annotations
from typing import MutableMapping, MutableSequence
import proto  # type: ignore
from cloudsdk.google.protobuf import duration_pb2  # type: ignore
from cloudsdk.google.protobuf import field_mask_pb2  # type: ignore
from cloudsdk.google.protobuf import timestamp_pb2  # type: ignore
from google.pubsub_v1.types import schema as gp_schema
class SubscriptionProperties(proto.Message):
    """Subscription properties sent as part of the response.

        Attributes:
            exactly_once_delivery_enabled (bool):
                True iff exactly once delivery is enabled for
                this subscription.
            message_ordering_enabled (bool):
                True iff message ordering is enabled for this
                subscription.
        """
    exactly_once_delivery_enabled: bool = proto.Field(proto.BOOL, number=1)
    message_ordering_enabled: bool = proto.Field(proto.BOOL, number=2)