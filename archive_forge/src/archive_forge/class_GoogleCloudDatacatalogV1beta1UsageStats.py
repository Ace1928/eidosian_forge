from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDatacatalogV1beta1UsageStats(_messages.Message):
    """Detailed counts on the entry's usage. Caveats: - Only BigQuery tables
  have usage stats - The usage stats only include BigQuery query jobs - The
  usage stats might be underestimated, e.g. wildcard table references are not
  yet counted in usage computation
  https://cloud.google.com/bigquery/docs/querying-wildcard-tables

  Fields:
    totalCancellations: The number of times that the underlying entry was
      attempted to be used but was cancelled by the user.
    totalCompletions: The number of times that the underlying entry was
      successfully used.
    totalExecutionTimeForCompletionsMillis: Total time spent (in milliseconds)
      during uses the resulted in completions.
    totalFailures: The number of times that the underlying entry was attempted
      to be used but failed.
  """
    totalCancellations = _messages.FloatField(1, variant=_messages.Variant.FLOAT)
    totalCompletions = _messages.FloatField(2, variant=_messages.Variant.FLOAT)
    totalExecutionTimeForCompletionsMillis = _messages.FloatField(3, variant=_messages.Variant.FLOAT)
    totalFailures = _messages.FloatField(4, variant=_messages.Variant.FLOAT)