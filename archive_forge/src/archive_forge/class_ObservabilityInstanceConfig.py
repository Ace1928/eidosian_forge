from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ObservabilityInstanceConfig(_messages.Message):
    """Observability Instance specific configuration.

  Fields:
    enabled: Observability feature status for an instance. This is a read-only
      flag and modifiable only by producer API. This flag is turned "off" by
      default.
    maxQueryStringLength: Query string length. The default value is 10k.
    preserveComments: Preserve comments in query string for an instance. This
      flag is turned "off" by default.
    queryPlansPerMinute: Number of query execution plans captured by Insights
      per minute for all queries combined. The default value is 5. Any integer
      between 0 to 20 is considered valid.
    recordApplicationTags: Record application tags for an instance. This flag
      is turned "off" by default.
    trackActiveQueries: Track actively running queries on the instance. If not
      set, this flag is "off" by default.
    trackWaitEventTypes: Output only. Track wait event types during query
      execution for an instance. This flag is turned "on" by default but
      tracking is enabled only after observability enabled flag is also turned
      on. This is read-only flag and only modifiable by producer API.
    trackWaitEvents: Track wait events during query execution for an instance.
      This flag is turned "on" by default but tracking is enabled only after
      observability enabled flag is also turned on.
  """
    enabled = _messages.BooleanField(1)
    maxQueryStringLength = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    preserveComments = _messages.BooleanField(3)
    queryPlansPerMinute = _messages.IntegerField(4, variant=_messages.Variant.INT32)
    recordApplicationTags = _messages.BooleanField(5)
    trackActiveQueries = _messages.BooleanField(6)
    trackWaitEventTypes = _messages.BooleanField(7)
    trackWaitEvents = _messages.BooleanField(8)