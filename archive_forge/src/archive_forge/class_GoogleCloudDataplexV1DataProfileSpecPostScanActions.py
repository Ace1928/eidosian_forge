from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDataplexV1DataProfileSpecPostScanActions(_messages.Message):
    """The configuration of post scan actions of DataProfileScan job.

  Fields:
    bigqueryExport: Optional. If set, results will be exported to the provided
      BigQuery table.
  """
    bigqueryExport = _messages.MessageField('GoogleCloudDataplexV1DataProfileSpecPostScanActionsBigQueryExport', 1)