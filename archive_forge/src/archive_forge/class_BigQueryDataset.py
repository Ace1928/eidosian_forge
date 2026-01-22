from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BigQueryDataset(_messages.Message):
    """Describes a BigQuery dataset that was created by a link.

  Fields:
    datasetId: Output only. The full resource name of the BigQuery dataset.
      The DATASET_ID will match the ID of the link, so the link must match the
      naming restrictions of BigQuery datasets (alphanumeric characters and
      underscores only).The dataset will have a resource path of
      "bigquery.googleapis.com/projects/PROJECT_ID/datasets/DATASET_ID"
  """
    datasetId = _messages.StringField(1)