from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AiplatformProjectsLocationsMetadataStoresContextsPatchRequest(_messages.Message):
    """A AiplatformProjectsLocationsMetadataStoresContextsPatchRequest object.

  Fields:
    allowMissing: If set to true, and the Context is not found, a new Context
      is created.
    googleCloudAiplatformV1Context: A GoogleCloudAiplatformV1Context resource
      to be passed as the request body.
    name: Immutable. The resource name of the Context.
    updateMask: Optional. A FieldMask indicating which fields should be
      updated.
  """
    allowMissing = _messages.BooleanField(1)
    googleCloudAiplatformV1Context = _messages.MessageField('GoogleCloudAiplatformV1Context', 2)
    name = _messages.StringField(3, required=True)
    updateMask = _messages.StringField(4)