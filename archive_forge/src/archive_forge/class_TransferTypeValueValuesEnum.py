from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TransferTypeValueValuesEnum(_messages.Enum):
    """Deprecated. This field has no effect.

    Values:
      TRANSFER_TYPE_UNSPECIFIED: Invalid or Unknown transfer type placeholder.
      BATCH: Batch data transfer.
      STREAMING: Streaming data transfer. Streaming data source currently
        doesn't support multiple transfer configs per project.
    """
    TRANSFER_TYPE_UNSPECIFIED = 0
    BATCH = 1
    STREAMING = 2