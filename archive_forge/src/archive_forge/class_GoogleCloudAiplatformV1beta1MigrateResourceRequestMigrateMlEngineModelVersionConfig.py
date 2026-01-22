from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1MigrateResourceRequestMigrateMlEngineModelVersionConfig(_messages.Message):
    """Config for migrating version in ml.googleapis.com to Vertex AI's Model.

  Fields:
    endpoint: Required. The ml.googleapis.com endpoint that this model version
      should be migrated from. Example values: * ml.googleapis.com * us-
      centrall-ml.googleapis.com * europe-west4-ml.googleapis.com * asia-
      east1-ml.googleapis.com
    modelDisplayName: Required. Display name of the model in Vertex AI. System
      will pick a display name if unspecified.
    modelVersion: Required. Full resource name of ml engine model version.
      Format: `projects/{project}/models/{model}/versions/{version}`.
  """
    endpoint = _messages.StringField(1)
    modelDisplayName = _messages.StringField(2)
    modelVersion = _messages.StringField(3)