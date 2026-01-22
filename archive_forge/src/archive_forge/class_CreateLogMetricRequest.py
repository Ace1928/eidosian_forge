from __future__ import annotations
from typing import MutableMapping, MutableSequence
import proto  # type: ignore
from google.api import distribution_pb2  # type: ignore
from google.api import metric_pb2  # type: ignore
from cloudsdk.google.protobuf import timestamp_pb2  # type: ignore
class CreateLogMetricRequest(proto.Message):
    """The parameters to CreateLogMetric.

    Attributes:
        parent (str):
            Required. The resource name of the project in which to
            create the metric:

            ::

                "projects/[PROJECT_ID]"

            The new metric must be provided in the request.
        metric (googlecloudsdk.generated_clients.gapic_clients.logging_v2.types.LogMetric):
            Required. The new logs-based metric, which
            must not have an identifier that already exists.
    """
    parent: str = proto.Field(proto.STRING, number=1)
    metric: 'LogMetric' = proto.Field(proto.MESSAGE, number=2, message='LogMetric')