from __future__ import annotations
from typing import MutableMapping, MutableSequence
import proto  # type: ignore
from cloudsdk.google.protobuf import field_mask_pb2  # type: ignore
from cloudsdk.google.protobuf import timestamp_pb2  # type: ignore
class GetCmekSettingsRequest(proto.Message):
    """The parameters to
    [GetCmekSettings][google.logging.v2.ConfigServiceV2.GetCmekSettings].

    See `Enabling CMEK for Log
    Router <https://cloud.google.com/logging/docs/routing/managed-encryption>`__
    for more information.

    Attributes:
        name (str):
            Required. The resource for which to retrieve CMEK settings.

            ::

                "projects/[PROJECT_ID]/cmekSettings"
                "organizations/[ORGANIZATION_ID]/cmekSettings"
                "billingAccounts/[BILLING_ACCOUNT_ID]/cmekSettings"
                "folders/[FOLDER_ID]/cmekSettings"

            For example:

            ``"organizations/12345/cmekSettings"``

            Note: CMEK for the Log Router can be configured for Google
            Cloud projects, folders, organizations, and billing
            accounts. Once configured for an organization, it applies to
            all projects and folders in the Google Cloud organization.
    """
    name: str = proto.Field(proto.STRING, number=1)