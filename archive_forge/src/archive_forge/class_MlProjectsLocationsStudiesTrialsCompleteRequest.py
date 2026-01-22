from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MlProjectsLocationsStudiesTrialsCompleteRequest(_messages.Message):
    """A MlProjectsLocationsStudiesTrialsCompleteRequest object.

  Fields:
    googleCloudMlV1CompleteTrialRequest: A GoogleCloudMlV1CompleteTrialRequest
      resource to be passed as the request body.
    name: Required. The trial name.metat
  """
    googleCloudMlV1CompleteTrialRequest = _messages.MessageField('GoogleCloudMlV1CompleteTrialRequest', 1)
    name = _messages.StringField(2, required=True)