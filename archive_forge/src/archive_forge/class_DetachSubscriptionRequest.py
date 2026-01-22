from __future__ import annotations
from typing import MutableMapping, MutableSequence
import proto  # type: ignore
from cloudsdk.google.protobuf import duration_pb2  # type: ignore
from cloudsdk.google.protobuf import field_mask_pb2  # type: ignore
from cloudsdk.google.protobuf import timestamp_pb2  # type: ignore
from google.pubsub_v1.types import schema as gp_schema
class DetachSubscriptionRequest(proto.Message):
    """Request for the DetachSubscription method.

    Attributes:
        subscription (str):
            Required. The subscription to detach. Format is
            ``projects/{project}/subscriptions/{subscription}``.
    """
    subscription: str = proto.Field(proto.STRING, number=1)