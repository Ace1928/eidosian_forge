from __future__ import annotations
from typing import MutableMapping, MutableSequence
import proto  # type: ignore
from cloudsdk.google.protobuf import duration_pb2  # type: ignore
from cloudsdk.google.protobuf import field_mask_pb2  # type: ignore
from cloudsdk.google.protobuf import timestamp_pb2  # type: ignore
from google.type import date_pb2  # type: ignore
class ListNotificationConfigsRequest(proto.Message):
    """Request message for ListNotifications.

    Attributes:
        parent (str):
            Required. Name of a Google Cloud Storage
            bucket.
        page_size (int):
            Optional. The maximum number of NotificationConfigs to
            return. The service may return fewer than this value. The
            default value is 100. Specifying a value above 100 will
            result in a page_size of 100.
        page_token (str):
            Optional. A page token, received from a previous
            ``ListNotificationConfigs`` call. Provide this to retrieve
            the subsequent page.

            When paginating, all other parameters provided to
            ``ListNotificationConfigs`` must match the call that
            provided the page token.
    """
    parent: str = proto.Field(proto.STRING, number=1)
    page_size: int = proto.Field(proto.INT32, number=2)
    page_token: str = proto.Field(proto.STRING, number=3)