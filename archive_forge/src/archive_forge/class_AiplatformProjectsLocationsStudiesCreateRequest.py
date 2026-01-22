from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AiplatformProjectsLocationsStudiesCreateRequest(_messages.Message):
    """A AiplatformProjectsLocationsStudiesCreateRequest object.

  Fields:
    googleCloudAiplatformV1Study: A GoogleCloudAiplatformV1Study resource to
      be passed as the request body.
    parent: Required. The resource name of the Location to create the
      CustomJob in. Format: `projects/{project}/locations/{location}`
  """
    googleCloudAiplatformV1Study = _messages.MessageField('GoogleCloudAiplatformV1Study', 1)
    parent = _messages.StringField(2, required=True)