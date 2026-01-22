from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1MigratableResourceAutomlDataset(_messages.Message):
    """Represents one Dataset in automl.googleapis.com.

  Fields:
    dataset: Full resource name of automl Dataset. Format:
      `projects/{project}/locations/{location}/datasets/{dataset}`.
    datasetDisplayName: The Dataset's display name in automl.googleapis.com.
  """
    dataset = _messages.StringField(1)
    datasetDisplayName = _messages.StringField(2)