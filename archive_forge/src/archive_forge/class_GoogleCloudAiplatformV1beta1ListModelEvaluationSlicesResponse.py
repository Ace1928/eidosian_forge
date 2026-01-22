from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1ListModelEvaluationSlicesResponse(_messages.Message):
    """Response message for ModelService.ListModelEvaluationSlices.

  Fields:
    modelEvaluationSlices: List of ModelEvaluations in the requested page.
    nextPageToken: A token to retrieve next page of results. Pass to
      ListModelEvaluationSlicesRequest.page_token to obtain that page.
  """
    modelEvaluationSlices = _messages.MessageField('GoogleCloudAiplatformV1beta1ModelEvaluationSlice', 1, repeated=True)
    nextPageToken = _messages.StringField(2)