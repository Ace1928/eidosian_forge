from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDatacatalogV1UsageStats(_messages.Message):
    """Detailed statistics on the entry's usage. Usage statistics have the
  following limitations: - Only BigQuery tables have them. - They only include
  BigQuery query jobs. - They might be underestimated because wildcard table
  references are not yet counted. For more information, see [Querying multiple
  tables using a wildcard table]
  (https://cloud.google.com/bigquery/docs/querying-wildcard-tables)

  Fields:
    totalCancellations: The number of cancelled attempts to use the underlying
      entry.
    totalCompletions: The number of successful uses of the underlying entry.
    totalExecutionTimeForCompletionsMillis: Total time spent only on
      successful uses, in milliseconds.
    totalFailures: The number of failed attempts to use the underlying entry.
  """
    totalCancellations = _messages.FloatField(1, variant=_messages.Variant.FLOAT)
    totalCompletions = _messages.FloatField(2, variant=_messages.Variant.FLOAT)
    totalExecutionTimeForCompletionsMillis = _messages.FloatField(3, variant=_messages.Variant.FLOAT)
    totalFailures = _messages.FloatField(4, variant=_messages.Variant.FLOAT)