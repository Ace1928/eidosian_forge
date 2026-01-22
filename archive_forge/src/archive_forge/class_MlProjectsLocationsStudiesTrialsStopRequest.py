from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MlProjectsLocationsStudiesTrialsStopRequest(_messages.Message):
    """A MlProjectsLocationsStudiesTrialsStopRequest object.

  Fields:
    googleCloudMlV1StopTrialRequest: A GoogleCloudMlV1StopTrialRequest
      resource to be passed as the request body.
    name: Required. The trial name.
  """
    googleCloudMlV1StopTrialRequest = _messages.MessageField('GoogleCloudMlV1StopTrialRequest', 1)
    name = _messages.StringField(2, required=True)