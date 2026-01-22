from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DataplexTransferStatusValueValuesEnum(_messages.Enum):
    """Optional. Transfer status of the TagTemplate

    Values:
      DATAPLEX_TRANSFER_STATUS_UNSPECIFIED: Default value. TagTemplate and its
        tags are only visible and editable in DataCatalog.
      MIGRATED: TagTemplate and its tags are auto-copied to Dataplex service.
        Visible in both services. Editable in DataCatalog, read-only in
        Dataplex.
    """
    DATAPLEX_TRANSFER_STATUS_UNSPECIFIED = 0
    MIGRATED = 1