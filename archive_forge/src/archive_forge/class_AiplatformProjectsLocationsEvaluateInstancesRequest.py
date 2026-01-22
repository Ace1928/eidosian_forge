from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AiplatformProjectsLocationsEvaluateInstancesRequest(_messages.Message):
    """A AiplatformProjectsLocationsEvaluateInstancesRequest object.

  Fields:
    googleCloudAiplatformV1beta1EvaluateInstancesRequest: A
      GoogleCloudAiplatformV1beta1EvaluateInstancesRequest resource to be
      passed as the request body.
    location: Required. The resource name of the Location to evaluate the
      instances. Format: `projects/{project}/locations/{location}`
  """
    googleCloudAiplatformV1beta1EvaluateInstancesRequest = _messages.MessageField('GoogleCloudAiplatformV1beta1EvaluateInstancesRequest', 1)
    location = _messages.StringField(2, required=True)