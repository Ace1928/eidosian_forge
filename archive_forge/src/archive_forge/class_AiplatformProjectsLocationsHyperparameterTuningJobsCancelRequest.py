from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AiplatformProjectsLocationsHyperparameterTuningJobsCancelRequest(_messages.Message):
    """A AiplatformProjectsLocationsHyperparameterTuningJobsCancelRequest
  object.

  Fields:
    googleCloudAiplatformV1CancelHyperparameterTuningJobRequest: A
      GoogleCloudAiplatformV1CancelHyperparameterTuningJobRequest resource to
      be passed as the request body.
    name: Required. The name of the HyperparameterTuningJob to cancel. Format:
      `projects/{project}/locations/{location}/hyperparameterTuningJobs/{hyper
      parameter_tuning_job}`
  """
    googleCloudAiplatformV1CancelHyperparameterTuningJobRequest = _messages.MessageField('GoogleCloudAiplatformV1CancelHyperparameterTuningJobRequest', 1)
    name = _messages.StringField(2, required=True)