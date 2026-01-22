from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AiplatformProjectsLocationsMetadataStoresExecutionsPatchRequest(_messages.Message):
    """A AiplatformProjectsLocationsMetadataStoresExecutionsPatchRequest
  object.

  Fields:
    allowMissing: If set to true, and the Execution is not found, a new
      Execution is created.
    googleCloudAiplatformV1Execution: A GoogleCloudAiplatformV1Execution
      resource to be passed as the request body.
    name: Output only. The resource name of the Execution.
    updateMask: Optional. A FieldMask indicating which fields should be
      updated.
  """
    allowMissing = _messages.BooleanField(1)
    googleCloudAiplatformV1Execution = _messages.MessageField('GoogleCloudAiplatformV1Execution', 2)
    name = _messages.StringField(3, required=True)
    updateMask = _messages.StringField(4)