from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1BigQueryDestination(_messages.Message):
    """The BigQuery location for the output content.

  Fields:
    outputUri: Required. BigQuery URI to a project or table, up to 2000
      characters long. When only the project is specified, the Dataset and
      Table is created. When the full table reference is specified, the
      Dataset must exist and table must not exist. Accepted forms: * BigQuery
      path. For example: `bq://projectId` or `bq://projectId.bqDatasetId` or
      `bq://projectId.bqDatasetId.bqTableId`.
  """
    outputUri = _messages.StringField(1)