from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AiplatformProjectsLocationsStudiesTrialsStopRequest(_messages.Message):
    """A AiplatformProjectsLocationsStudiesTrialsStopRequest object.

  Fields:
    googleCloudAiplatformV1StopTrialRequest: A
      GoogleCloudAiplatformV1StopTrialRequest resource to be passed as the
      request body.
    name: Required. The Trial's name. Format:
      `projects/{project}/locations/{location}/studies/{study}/trials/{trial}`
  """
    googleCloudAiplatformV1StopTrialRequest = _messages.MessageField('GoogleCloudAiplatformV1StopTrialRequest', 1)
    name = _messages.StringField(2, required=True)