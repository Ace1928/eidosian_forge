from __future__ import annotations
from typing import MutableMapping, MutableSequence
import proto  # type: ignore
from cloudsdk.google.protobuf import struct_pb2  # type: ignore
Recommendation to add new indexes to run queries more
        efficiently.

        Attributes:
            ddl (MutableSequence[str]):
                Optional. DDL statements to add new indexes
                that will improve the query.
            improvement_factor (float):
                Optional. Estimated latency improvement
                factor. For example if the query currently takes
                500 ms to run and the estimated latency with new
                indexes is 100 ms this field will be 5.
        