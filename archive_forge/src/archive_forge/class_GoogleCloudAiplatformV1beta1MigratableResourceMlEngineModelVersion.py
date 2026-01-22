from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1MigratableResourceMlEngineModelVersion(_messages.Message):
    """Represents one model Version in ml.googleapis.com.

  Fields:
    endpoint: The ml.googleapis.com endpoint that this model Version currently
      lives in. Example values: * ml.googleapis.com * us-centrall-
      ml.googleapis.com * europe-west4-ml.googleapis.com * asia-
      east1-ml.googleapis.com
    version: Full resource name of ml engine model Version. Format:
      `projects/{project}/models/{model}/versions/{version}`.
  """
    endpoint = _messages.StringField(1)
    version = _messages.StringField(2)