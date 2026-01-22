from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1MigratableResourceAutomlModel(_messages.Message):
    """Represents one Model in automl.googleapis.com.

  Fields:
    model: Full resource name of automl Model. Format:
      `projects/{project}/locations/{location}/models/{model}`.
    modelDisplayName: The Model's display name in automl.googleapis.com.
  """
    model = _messages.StringField(1)
    modelDisplayName = _messages.StringField(2)