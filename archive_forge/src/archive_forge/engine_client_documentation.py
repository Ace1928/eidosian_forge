import datetime
import sys
from typing import (
import warnings
import duet
import proto
from google.api_core.exceptions import GoogleAPICallError, NotFound
from google.protobuf import any_pb2, field_mask_pb2
from google.protobuf.timestamp_pb2 import Timestamp
from cirq._compat import cached_property
from cirq._compat import deprecated_parameter
from cirq_google.cloud import quantum
from cirq_google.engine.asyncio_executor import AsyncioExecutor
from cirq_google.engine import stream_manager
Returns a list of quantum time slots on a processor.

        Args:
            project_id: A project_id of the parent Google Cloud Project.
            processor_id: The processor unique identifier.
            filter_str:  A string expression for filtering the quantum
                time slots returned by the list command. The fields
                eligible for filtering are `start_time`, `end_time`.

        Returns:
            A list of QuantumTimeSlot objects.
        