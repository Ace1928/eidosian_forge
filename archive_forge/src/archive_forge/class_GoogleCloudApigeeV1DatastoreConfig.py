from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudApigeeV1DatastoreConfig(_messages.Message):
    """Configuration detail for datastore

  Fields:
    bucketName: Name of the Cloud Storage bucket. Required for `gcs`
      target_type.
    datasetName: BigQuery dataset name Required for `bigquery` target_type.
    path: Path of Cloud Storage bucket Required for `gcs` target_type.
    projectId: Required. GCP project in which the datastore exists
    tablePrefix: Prefix of BigQuery table Required for `bigquery` target_type.
  """
    bucketName = _messages.StringField(1)
    datasetName = _messages.StringField(2)
    path = _messages.StringField(3)
    projectId = _messages.StringField(4)
    tablePrefix = _messages.StringField(5)