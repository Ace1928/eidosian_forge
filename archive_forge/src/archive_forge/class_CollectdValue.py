from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CollectdValue(_messages.Message):
    """A single data point from a collectd-based plugin.

  Enums:
    DataSourceTypeValueValuesEnum: The type of measurement.

  Fields:
    dataSourceName: The data source for the collectd value. For example, there
      are two data sources for network measurements: "rx" and "tx".
    dataSourceType: The type of measurement.
    value: The measurement value.
  """

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
    dataSourceName = _messages.StringField(1)
    dataSourceType = _messages.EnumField('DataSourceTypeValueValuesEnum', 2)
    value = _messages.MessageField('TypedValue', 3)