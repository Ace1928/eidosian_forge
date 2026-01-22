from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DataSourceTypeValueValuesEnum(_messages.Enum):
    """The type of measurement.

    Values:
      UNSPECIFIED_DATA_SOURCE_TYPE: An unspecified data source type. This
        corresponds to
        google.api.MetricDescriptor.MetricKind.METRIC_KIND_UNSPECIFIED.
      GAUGE: An instantaneous measurement of a varying quantity. This
        corresponds to google.api.MetricDescriptor.MetricKind.GAUGE.
      COUNTER: A cumulative value over time. This corresponds to
        google.api.MetricDescriptor.MetricKind.CUMULATIVE.
      DERIVE: A rate of change of the measurement.
      ABSOLUTE: An amount of change since the last measurement interval. This
        corresponds to google.api.MetricDescriptor.MetricKind.DELTA.
    """
    UNSPECIFIED_DATA_SOURCE_TYPE = 0
    GAUGE = 1
    COUNTER = 2
    DERIVE = 3
    ABSOLUTE = 4