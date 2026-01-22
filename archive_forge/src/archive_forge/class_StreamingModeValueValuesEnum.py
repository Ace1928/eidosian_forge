from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class StreamingModeValueValuesEnum(_messages.Enum):
    """Optional. Specifies the Streaming Engine message processing
    guarantees. Reduces cost and latency but might result in duplicate
    messages committed to storage. Designed to run simple mapping streaming
    ETL jobs at the lowest cost. For example, Change Data Capture (CDC) to
    BigQuery is a canonical use case. For more information, see [Set the
    pipeline streaming
    mode](https://cloud.google.com/dataflow/docs/guides/streaming-modes).

    Values:
      STREAMING_MODE_UNSPECIFIED: Run in the default mode.
      STREAMING_MODE_EXACTLY_ONCE: In this mode, message deduplication is
        performed against persistent state to make sure each message is
        processed and committed to storage exactly once.
      STREAMING_MODE_AT_LEAST_ONCE: Message deduplication is not performed.
        Messages might be processed multiple times, and the results are
        applied multiple times. Note: Setting this value also enables
        Streaming Engine and Streaming Engine resource-based billing.
    """
    STREAMING_MODE_UNSPECIFIED = 0
    STREAMING_MODE_EXACTLY_ONCE = 1
    STREAMING_MODE_AT_LEAST_ONCE = 2