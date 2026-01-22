from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDataplexV1DataScanEventPostScanActionsResult(_messages.Message):
    """Post scan actions result for data scan job.

  Fields:
    bigqueryExportResult: The result of BigQuery export post scan action.
  """
    bigqueryExportResult = _messages.MessageField('GoogleCloudDataplexV1DataScanEventPostScanActionsResultBigQueryExportResult', 1)