from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GooglePrivacyDlpV2TableLocation(_messages.Message):
    """Location of a finding within a table.

  Fields:
    rowIndex: The zero-based index of the row where the finding is located.
      Only populated for resources that have a natural ordering, not BigQuery.
      In BigQuery, to identify the row a finding came from, populate
      BigQueryOptions.identifying_fields with your primary key column names
      and when you store the findings the value of those columns will be
      stored inside of Finding.
  """
    rowIndex = _messages.IntegerField(1)