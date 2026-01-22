from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1MigrateResourceRequest(_messages.Message):
    """Config of migrating one resource from automl.googleapis.com,
  datalabeling.googleapis.com and ml.googleapis.com to Vertex AI.

  Fields:
    migrateAutomlDatasetConfig: Config for migrating Dataset in
      automl.googleapis.com to Vertex AI's Dataset.
    migrateAutomlModelConfig: Config for migrating Model in
      automl.googleapis.com to Vertex AI's Model.
    migrateDataLabelingDatasetConfig: Config for migrating Dataset in
      datalabeling.googleapis.com to Vertex AI's Dataset.
    migrateMlEngineModelVersionConfig: Config for migrating Version in
      ml.googleapis.com to Vertex AI's Model.
  """
    migrateAutomlDatasetConfig = _messages.MessageField('GoogleCloudAiplatformV1beta1MigrateResourceRequestMigrateAutomlDatasetConfig', 1)
    migrateAutomlModelConfig = _messages.MessageField('GoogleCloudAiplatformV1beta1MigrateResourceRequestMigrateAutomlModelConfig', 2)
    migrateDataLabelingDatasetConfig = _messages.MessageField('GoogleCloudAiplatformV1beta1MigrateResourceRequestMigrateDataLabelingDatasetConfig', 3)
    migrateMlEngineModelVersionConfig = _messages.MessageField('GoogleCloudAiplatformV1beta1MigrateResourceRequestMigrateMlEngineModelVersionConfig', 4)