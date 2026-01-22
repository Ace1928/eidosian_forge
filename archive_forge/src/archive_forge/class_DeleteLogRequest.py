from __future__ import annotations
from typing import MutableMapping, MutableSequence
import proto  # type: ignore
from google.api import monitored_resource_pb2  # type: ignore
from cloudsdk.google.protobuf import duration_pb2  # type: ignore
from google.rpc import status_pb2  # type: ignore
from googlecloudsdk.generated_clients.gapic_clients.logging_v2.types import log_entry
class DeleteLogRequest(proto.Message):
    """The parameters to DeleteLog.

    Attributes:
        log_name (str):
            Required. The resource name of the log to delete:

            -  ``projects/[PROJECT_ID]/logs/[LOG_ID]``
            -  ``organizations/[ORGANIZATION_ID]/logs/[LOG_ID]``
            -  ``billingAccounts/[BILLING_ACCOUNT_ID]/logs/[LOG_ID]``
            -  ``folders/[FOLDER_ID]/logs/[LOG_ID]``

            ``[LOG_ID]`` must be URL-encoded. For example,
            ``"projects/my-project-id/logs/syslog"``,
            ``"organizations/123/logs/cloudaudit.googleapis.com%2Factivity"``.

            For more information about log names, see
            [LogEntry][google.logging.v2.LogEntry].
    """
    log_name: str = proto.Field(proto.STRING, number=1)