from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class InsightsConfig(_messages.Message):
    """Insights configuration. This specifies when Cloud SQL Insights feature
  is enabled and optional configuration.

  Fields:
    queryInsightsEnabled: Whether Query Insights feature is enabled.
    queryPlansPerMinute: Number of query execution plans captured by Insights
      per minute for all queries combined. Default is 5.
    queryStringLength: Maximum query length stored in bytes. Default value:
      1024 bytes. Range: 256-4500 bytes. Query length more than this field
      value will be truncated to this value. When unset, query length will be
      the default value. Changing query length will restart the database.
    recordApplicationTags: Whether Query Insights will record application tags
      from query when enabled.
    recordClientAddress: Whether Query Insights will record client address
      when enabled.
  """
    queryInsightsEnabled = _messages.BooleanField(1)
    queryPlansPerMinute = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    queryStringLength = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    recordApplicationTags = _messages.BooleanField(4)
    recordClientAddress = _messages.BooleanField(5)