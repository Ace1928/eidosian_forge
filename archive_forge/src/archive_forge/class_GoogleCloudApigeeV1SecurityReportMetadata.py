from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudApigeeV1SecurityReportMetadata(_messages.Message):
    """Metadata for the security report.

  Fields:
    dimensions: Dimensions of the SecurityReport.
    endTimestamp: End timestamp of the query range.
    metrics: Metrics of the SecurityReport. Example:
      ["name:bot_count,func:sum,alias:sum_bot_count"]
    mimeType: MIME type / Output format.
    startTimestamp: Start timestamp of the query range.
    timeUnit: Query GroupBy time unit. Example: "seconds", "minute", "hour"
  """
    dimensions = _messages.StringField(1, repeated=True)
    endTimestamp = _messages.StringField(2)
    metrics = _messages.StringField(3, repeated=True)
    mimeType = _messages.StringField(4)
    startTimestamp = _messages.StringField(5)
    timeUnit = _messages.StringField(6)