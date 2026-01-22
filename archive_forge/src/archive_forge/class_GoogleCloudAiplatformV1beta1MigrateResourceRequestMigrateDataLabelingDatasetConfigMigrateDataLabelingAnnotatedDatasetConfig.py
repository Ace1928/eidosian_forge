from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1MigrateResourceRequestMigrateDataLabelingDatasetConfigMigrateDataLabelingAnnotatedDatasetConfig(_messages.Message):
    """Config for migrating AnnotatedDataset in datalabeling.googleapis.com to
  Vertex AI's SavedQuery.

  Fields:
    annotatedDataset: Required. Full resource name of data labeling
      AnnotatedDataset. Format: `projects/{project}/datasets/{dataset}/annotat
      edDatasets/{annotated_dataset}`.
  """
    annotatedDataset = _messages.StringField(1)