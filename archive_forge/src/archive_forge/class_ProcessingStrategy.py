from __future__ import annotations
from typing import MutableMapping, MutableSequence
from google.protobuf import duration_pb2  # type: ignore
from google.protobuf import field_mask_pb2  # type: ignore
from google.protobuf import timestamp_pb2  # type: ignore
from google.rpc import status_pb2  # type: ignore
import proto  # type: ignore
class ProcessingStrategy(proto.Enum):
    """Possible processing strategies for batch requests.

        Values:
            PROCESSING_STRATEGY_UNSPECIFIED (0):
                Default value for the processing strategy.
                The request is processed as soon as its
                received.
            DYNAMIC_BATCHING (1):
                If selected, processes the request during
                lower utilization periods for a price discount.
                The request is fulfilled within 24 hours.
        """
    PROCESSING_STRATEGY_UNSPECIFIED = 0
    DYNAMIC_BATCHING = 1