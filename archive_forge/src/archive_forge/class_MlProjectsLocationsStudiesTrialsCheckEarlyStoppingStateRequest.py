from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MlProjectsLocationsStudiesTrialsCheckEarlyStoppingStateRequest(_messages.Message):
    """A MlProjectsLocationsStudiesTrialsCheckEarlyStoppingStateRequest object.

  Fields:
    googleCloudMlV1CheckTrialEarlyStoppingStateRequest: A
      GoogleCloudMlV1CheckTrialEarlyStoppingStateRequest resource to be passed
      as the request body.
    name: Required. The trial name.
  """
    googleCloudMlV1CheckTrialEarlyStoppingStateRequest = _messages.MessageField('GoogleCloudMlV1CheckTrialEarlyStoppingStateRequest', 1)
    name = _messages.StringField(2, required=True)