from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CreateCollectdTimeSeriesRequest(_messages.Message):
    """The CreateCollectdTimeSeries request.

  Fields:
    collectdPayloads: The collectd payloads representing the time series data.
      You must not include more than a single point for each time series, so
      no two payloads can have the same values for all of the fields plugin,
      plugin_instance, type, and type_instance.
    collectdVersion: The version of collectd that collected the data. Example:
      "5.3.0-192.el6".
    resource: The monitored resource associated with the time series.
  """
    collectdPayloads = _messages.MessageField('CollectdPayload', 1, repeated=True)
    collectdVersion = _messages.StringField(2)
    resource = _messages.MessageField('MonitoredResource', 3)