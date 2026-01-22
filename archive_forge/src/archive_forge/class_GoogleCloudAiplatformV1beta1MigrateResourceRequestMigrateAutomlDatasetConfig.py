from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1MigrateResourceRequestMigrateAutomlDatasetConfig(_messages.Message):
    """Config for migrating Dataset in automl.googleapis.com to Vertex AI's
  Dataset.

  Fields:
    dataset: Required. Full resource name of automl Dataset. Format:
      `projects/{project}/locations/{location}/datasets/{dataset}`.
    datasetDisplayName: Required. Display name of the Dataset in Vertex AI.
      System will pick a display name if unspecified.
  """
    dataset = _messages.StringField(1)
    datasetDisplayName = _messages.StringField(2)