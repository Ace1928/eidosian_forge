from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GooglePrivacyDlpV2DataProfileBigQueryRowSchema(_messages.Message):
    """The schema of data to be saved to the BigQuery table when the
  `DataProfileAction` is enabled.

  Fields:
    columnProfile: Column data profile column
    tableProfile: Table data profile column
  """
    columnProfile = _messages.MessageField('GooglePrivacyDlpV2ColumnDataProfile', 1)
    tableProfile = _messages.MessageField('GooglePrivacyDlpV2TableDataProfile', 2)