from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AiplatformProjectsLocationsStudiesTrialsAddTrialMeasurementRequest(_messages.Message):
    """A AiplatformProjectsLocationsStudiesTrialsAddTrialMeasurementRequest
  object.

  Fields:
    googleCloudAiplatformV1AddTrialMeasurementRequest: A
      GoogleCloudAiplatformV1AddTrialMeasurementRequest resource to be passed
      as the request body.
    trialName: Required. The name of the trial to add measurement. Format:
      `projects/{project}/locations/{location}/studies/{study}/trials/{trial}`
  """
    googleCloudAiplatformV1AddTrialMeasurementRequest = _messages.MessageField('GoogleCloudAiplatformV1AddTrialMeasurementRequest', 1)
    trialName = _messages.StringField(2, required=True)