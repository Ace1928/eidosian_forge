from __future__ import annotations
from typing import MutableMapping, MutableSequence
import proto  # type: ignore
from cloudsdk.google.protobuf import field_mask_pb2  # type: ignore
from cloudsdk.google.protobuf import timestamp_pb2  # type: ignore
class UpdateSettingsRequest(proto.Message):
    """The parameters to
    [UpdateSettings][google.logging.v2.ConfigServiceV2.UpdateSettings].

    See [Configure default settings for organizations and folders]
    (https://cloud.google.com/logging/docs/default-settings) for more
    information.

    Attributes:
        name (str):
            Required. The resource name for the settings to update.

            ::

                "organizations/[ORGANIZATION_ID]/settings"

            For example:

            ``"organizations/12345/settings"``
        settings (googlecloudsdk.generated_clients.gapic_clients.logging_v2.types.Settings):
            Required. The settings to update.

            See `Enabling CMEK for Log
            Router <https://cloud.google.com/logging/docs/routing/managed-encryption>`__
            for more information.
        update_mask (google.protobuf.field_mask_pb2.FieldMask):
            Optional. Field mask identifying which fields from
            ``settings`` should be updated. A field will be overwritten
            if and only if it is in the update mask. Output only fields
            cannot be updated.

            See [FieldMask][google.protobuf.FieldMask] for more
            information.

            For example: ``"updateMask=kmsKeyName"``
    """
    name: str = proto.Field(proto.STRING, number=1)
    settings: 'Settings' = proto.Field(proto.MESSAGE, number=2, message='Settings')
    update_mask: field_mask_pb2.FieldMask = proto.Field(proto.MESSAGE, number=3, message=field_mask_pb2.FieldMask)