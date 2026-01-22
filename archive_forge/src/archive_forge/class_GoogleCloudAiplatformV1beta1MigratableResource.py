from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1MigratableResource(_messages.Message):
    """Represents one resource that exists in automl.googleapis.com,
  datalabeling.googleapis.com or ml.googleapis.com.

  Fields:
    automlDataset: Output only. Represents one Dataset in
      automl.googleapis.com.
    automlModel: Output only. Represents one Model in automl.googleapis.com.
    dataLabelingDataset: Output only. Represents one Dataset in
      datalabeling.googleapis.com.
    lastMigrateTime: Output only. Timestamp when the last migration attempt on
      this MigratableResource started. Will not be set if there's no migration
      attempt on this MigratableResource.
    lastUpdateTime: Output only. Timestamp when this MigratableResource was
      last updated.
    mlEngineModelVersion: Output only. Represents one Version in
      ml.googleapis.com.
  """
    automlDataset = _messages.MessageField('GoogleCloudAiplatformV1beta1MigratableResourceAutomlDataset', 1)
    automlModel = _messages.MessageField('GoogleCloudAiplatformV1beta1MigratableResourceAutomlModel', 2)
    dataLabelingDataset = _messages.MessageField('GoogleCloudAiplatformV1beta1MigratableResourceDataLabelingDataset', 3)
    lastMigrateTime = _messages.StringField(4)
    lastUpdateTime = _messages.StringField(5)
    mlEngineModelVersion = _messages.MessageField('GoogleCloudAiplatformV1beta1MigratableResourceMlEngineModelVersion', 6)