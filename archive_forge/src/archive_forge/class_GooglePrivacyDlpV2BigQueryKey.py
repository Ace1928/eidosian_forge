from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GooglePrivacyDlpV2BigQueryKey(_messages.Message):
    """Row key for identifying a record in BigQuery table.

  Fields:
    rowNumber: Row number inferred at the time the table was scanned. This
      value is nondeterministic, cannot be queried, and may be null for
      inspection jobs. To locate findings within a table, specify
      `inspect_job.storage_config.big_query_options.identifying_fields` in
      `CreateDlpJobRequest`.
    tableReference: Complete BigQuery table reference.
  """
    rowNumber = _messages.IntegerField(1)
    tableReference = _messages.MessageField('GooglePrivacyDlpV2BigQueryTable', 2)