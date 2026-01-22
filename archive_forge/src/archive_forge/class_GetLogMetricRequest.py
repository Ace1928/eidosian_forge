from __future__ import annotations
from typing import MutableMapping, MutableSequence
import proto  # type: ignore
from google.api import distribution_pb2  # type: ignore
from google.api import metric_pb2  # type: ignore
from cloudsdk.google.protobuf import timestamp_pb2  # type: ignore
class GetLogMetricRequest(proto.Message):
    """The parameters to GetLogMetric.

    Attributes:
        metric_name (str):
            Required. The resource name of the desired metric:

            ::

                "projects/[PROJECT_ID]/metrics/[METRIC_ID]".
    """
    metric_name: str = proto.Field(proto.STRING, number=1)