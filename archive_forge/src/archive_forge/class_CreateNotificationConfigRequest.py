from __future__ import annotations
from typing import MutableMapping, MutableSequence
import proto  # type: ignore
from cloudsdk.google.protobuf import duration_pb2  # type: ignore
from cloudsdk.google.protobuf import field_mask_pb2  # type: ignore
from cloudsdk.google.protobuf import timestamp_pb2  # type: ignore
from google.type import date_pb2  # type: ignore
class CreateNotificationConfigRequest(proto.Message):
    """Request message for CreateNotificationConfig.

    Attributes:
        parent (str):
            Required. The bucket to which this
            NotificationConfig belongs.
        notification_config (googlecloudsdk.generated_clients.gapic_clients.storage_v2.types.NotificationConfig):
            Required. Properties of the
            NotificationConfig to be inserted.
    """
    parent: str = proto.Field(proto.STRING, number=1)
    notification_config: 'NotificationConfig' = proto.Field(proto.MESSAGE, number=2, message='NotificationConfig')