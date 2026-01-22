from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RecentQuery(_messages.Message):
    """Describes a recent query executed on the Logs Explorer or Log Analytics
  page within the last ~ 30 days.

  Fields:
    lastRunTime: Output only. The timestamp when this query was last run.
    loggingQuery: Logging query that can be executed in Logs Explorer or via
      Logging API.
    name: Output only. Resource name of the recent query.In the format:
      "projects/[PROJECT_ID]/locations/[LOCATION_ID]/recentQueries/[QUERY_ID]"
      For a list of supported locations, see Supported Regions
      (https://cloud.google.com/logging/docs/region-support)The QUERY_ID is a
      system generated alphanumeric ID.
    opsAnalyticsQuery: Analytics query that can be executed in Log Analytics.
  """
    lastRunTime = _messages.StringField(1)
    loggingQuery = _messages.MessageField('LoggingQuery', 2)
    name = _messages.StringField(3)
    opsAnalyticsQuery = _messages.MessageField('OpsAnalyticsQuery', 4)