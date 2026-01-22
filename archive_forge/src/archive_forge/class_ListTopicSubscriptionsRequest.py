from __future__ import annotations
from typing import MutableMapping, MutableSequence
import proto  # type: ignore
from google.cloud.pubsublite_v1.types import common
from cloudsdk.google.protobuf import field_mask_pb2  # type: ignore
from cloudsdk.google.protobuf import timestamp_pb2  # type: ignore
class ListTopicSubscriptionsRequest(proto.Message):
    """Request for ListTopicSubscriptions.

    Attributes:
        name (str):
            Required. The name of the topic whose
            subscriptions to list.
        page_size (int):
            The maximum number of subscriptions to
            return. The service may return fewer than this
            value. If unset or zero, all subscriptions for
            the given topic will be returned.
        page_token (str):
            A page token, received from a previous
            ``ListTopicSubscriptions`` call. Provide this to retrieve
            the subsequent page.

            When paginating, all other parameters provided to
            ``ListTopicSubscriptions`` must match the call that provided
            the page token.
    """
    name: str = proto.Field(proto.STRING, number=1)
    page_size: int = proto.Field(proto.INT32, number=2)
    page_token: str = proto.Field(proto.STRING, number=3)