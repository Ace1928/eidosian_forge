from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AiplatformProjectsLocationsSpecialistPoolsPatchRequest(_messages.Message):
    """A AiplatformProjectsLocationsSpecialistPoolsPatchRequest object.

  Fields:
    googleCloudAiplatformV1SpecialistPool: A
      GoogleCloudAiplatformV1SpecialistPool resource to be passed as the
      request body.
    name: Required. The resource name of the SpecialistPool.
    updateMask: Required. The update mask applies to the resource.
  """
    googleCloudAiplatformV1SpecialistPool = _messages.MessageField('GoogleCloudAiplatformV1SpecialistPool', 1)
    name = _messages.StringField(2, required=True)
    updateMask = _messages.StringField(3)