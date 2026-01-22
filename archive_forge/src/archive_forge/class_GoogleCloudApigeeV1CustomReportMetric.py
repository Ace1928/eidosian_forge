from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudApigeeV1CustomReportMetric(_messages.Message):
    """This encapsulates a metric property of the form sum(message_count) where
  name is message_count and function is sum

  Fields:
    function: aggregate function
    name: name of the metric
  """
    function = _messages.StringField(1)
    name = _messages.StringField(2)