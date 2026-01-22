from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AiplatformProjectsLocationsDataLabelingJobsCancelRequest(_messages.Message):
    """A AiplatformProjectsLocationsDataLabelingJobsCancelRequest object.

  Fields:
    googleCloudAiplatformV1CancelDataLabelingJobRequest: A
      GoogleCloudAiplatformV1CancelDataLabelingJobRequest resource to be
      passed as the request body.
    name: Required. The name of the DataLabelingJob. Format: `projects/{projec
      t}/locations/{location}/dataLabelingJobs/{data_labeling_job}`
  """
    googleCloudAiplatformV1CancelDataLabelingJobRequest = _messages.MessageField('GoogleCloudAiplatformV1CancelDataLabelingJobRequest', 1)
    name = _messages.StringField(2, required=True)