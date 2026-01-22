from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MlProjectsLocationsStudiesTrialsSuggestRequest(_messages.Message):
    """A MlProjectsLocationsStudiesTrialsSuggestRequest object.

  Fields:
    googleCloudMlV1SuggestTrialsRequest: A GoogleCloudMlV1SuggestTrialsRequest
      resource to be passed as the request body.
    parent: Required. The name of the study that the trial belongs to.
  """
    googleCloudMlV1SuggestTrialsRequest = _messages.MessageField('GoogleCloudMlV1SuggestTrialsRequest', 1)
    parent = _messages.StringField(2, required=True)