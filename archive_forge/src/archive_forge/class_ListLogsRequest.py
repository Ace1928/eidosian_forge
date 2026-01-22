from __future__ import annotations
from typing import MutableMapping, MutableSequence
import proto  # type: ignore
from google.api import monitored_resource_pb2  # type: ignore
from cloudsdk.google.protobuf import duration_pb2  # type: ignore
from google.rpc import status_pb2  # type: ignore
from googlecloudsdk.generated_clients.gapic_clients.logging_v2.types import log_entry
class ListLogsRequest(proto.Message):
    """The parameters to ListLogs.

    Attributes:
        parent (str):
            Required. The resource name to list logs for:

            -  ``projects/[PROJECT_ID]``
            -  ``organizations/[ORGANIZATION_ID]``
            -  ``billingAccounts/[BILLING_ACCOUNT_ID]``
            -  ``folders/[FOLDER_ID]``
        resource_names (MutableSequence[str]):
            Optional. List of resource names to list logs for:

            -  ``projects/[PROJECT_ID]/locations/[LOCATION_ID]/buckets/[BUCKET_ID]/views/[VIEW_ID]``
            -  ``organizations/[ORGANIZATION_ID]/locations/[LOCATION_ID]/buckets/[BUCKET_ID]/views/[VIEW_ID]``
            -  ``billingAccounts/[BILLING_ACCOUNT_ID]/locations/[LOCATION_ID]/buckets/[BUCKET_ID]/views/[VIEW_ID]``
            -  ``folders/[FOLDER_ID]/locations/[LOCATION_ID]/buckets/[BUCKET_ID]/views/[VIEW_ID]``

            To support legacy queries, it could also be:

            -  ``projects/[PROJECT_ID]``
            -  ``organizations/[ORGANIZATION_ID]``
            -  ``billingAccounts/[BILLING_ACCOUNT_ID]``
            -  ``folders/[FOLDER_ID]``

            The resource name in the ``parent`` field is added to this
            list.
        page_size (int):
            Optional. The maximum number of results to return from this
            request. Non-positive values are ignored. The presence of
            ``nextPageToken`` in the response indicates that more
            results might be available.
        page_token (str):
            Optional. If present, then retrieve the next batch of
            results from the preceding call to this method.
            ``pageToken`` must be the value of ``nextPageToken`` from
            the previous response. The values of other method parameters
            should be identical to those in the previous call.
    """
    parent: str = proto.Field(proto.STRING, number=1)
    resource_names: MutableSequence[str] = proto.RepeatedField(proto.STRING, number=8)
    page_size: int = proto.Field(proto.INT32, number=2)
    page_token: str = proto.Field(proto.STRING, number=3)