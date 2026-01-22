from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1ListTensorboardExperimentsResponse(_messages.Message):
    """Response message for TensorboardService.ListTensorboardExperiments.

  Fields:
    nextPageToken: A token, which can be sent as
      ListTensorboardExperimentsRequest.page_token to retrieve the next page.
      If this field is omitted, there are no subsequent pages.
    tensorboardExperiments: The TensorboardExperiments mathching the request.
  """
    nextPageToken = _messages.StringField(1)
    tensorboardExperiments = _messages.MessageField('GoogleCloudAiplatformV1beta1TensorboardExperiment', 2, repeated=True)