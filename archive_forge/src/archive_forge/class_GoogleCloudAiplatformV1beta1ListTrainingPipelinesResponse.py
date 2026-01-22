from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1ListTrainingPipelinesResponse(_messages.Message):
    """Response message for PipelineService.ListTrainingPipelines

  Fields:
    nextPageToken: A token to retrieve the next page of results. Pass to
      ListTrainingPipelinesRequest.page_token to obtain that page.
    trainingPipelines: List of TrainingPipelines in the requested page.
  """
    nextPageToken = _messages.StringField(1)
    trainingPipelines = _messages.MessageField('GoogleCloudAiplatformV1beta1TrainingPipeline', 2, repeated=True)