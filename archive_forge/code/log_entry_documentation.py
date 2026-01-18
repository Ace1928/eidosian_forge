from __future__ import annotations
from typing import MutableMapping, MutableSequence
import proto  # type: ignore
from google.api import monitored_resource_pb2  # type: ignore
from google.logging.type import http_request_pb2  # type: ignore
from google.logging.type import log_severity_pb2  # type: ignore
from cloudsdk.google.protobuf import any_pb2  # type: ignore
from cloudsdk.google.protobuf import struct_pb2  # type: ignore
from cloudsdk.google.protobuf import timestamp_pb2  # type: ignore
Additional information used to correlate multiple log
    entries. Used when a single LogEntry would exceed the Google
    Cloud Logging size limit and is split across multiple log
    entries.

    Attributes:
        uid (str):
            A globally unique identifier for all log entries in a
            sequence of split log entries. All log entries with the same
            \|LogSplit.uid\| are assumed to be part of the same sequence
            of split log entries.
        index (int):
            The index of this LogEntry in the sequence of split log
            entries. Log entries are given \|index\| values 0, 1, ...,
            n-1 for a sequence of n log entries.
        total_splits (int):
            The total number of log entries that the
            original LogEntry was split into.
    