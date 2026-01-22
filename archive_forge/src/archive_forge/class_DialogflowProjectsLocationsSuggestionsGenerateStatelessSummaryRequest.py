from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DialogflowProjectsLocationsSuggestionsGenerateStatelessSummaryRequest(_messages.Message):
    """A DialogflowProjectsLocationsSuggestionsGenerateStatelessSummaryRequest
  object.

  Fields:
    googleCloudDialogflowV2GenerateStatelessSummaryRequest: A
      GoogleCloudDialogflowV2GenerateStatelessSummaryRequest resource to be
      passed as the request body.
    parent: Required. The parent resource to charge for the Summary's
      generation. Format: `projects//locations/`.
  """
    googleCloudDialogflowV2GenerateStatelessSummaryRequest = _messages.MessageField('GoogleCloudDialogflowV2GenerateStatelessSummaryRequest', 1)
    parent = _messages.StringField(2, required=True)