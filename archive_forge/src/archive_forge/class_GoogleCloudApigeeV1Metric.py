from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudApigeeV1Metric(_messages.Message):
    """Encapsulates the metric data point. For example: ```{ "name":
  "sum(message_count)", "values" : [ { "timestamp": 1549004400000, "value":
  "39.0" }, { "timestamp" : 1548997200000, "value" : "0.0" } ] }``` or ```{
  "name": "sum(message_count)", "values" : ["39.0"] }```

  Fields:
    name: Metric name.
    values: List of metric values. Possible value formats include:
      `"values":["39.0"]` or `"values":[ { "value": "39.0", "timestamp":
      1232434354} ]`
  """
    name = _messages.StringField(1)
    values = _messages.MessageField('extra_types.JsonValue', 2, repeated=True)