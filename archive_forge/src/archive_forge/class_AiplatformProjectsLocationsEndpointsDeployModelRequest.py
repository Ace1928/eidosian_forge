from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AiplatformProjectsLocationsEndpointsDeployModelRequest(_messages.Message):
    """A AiplatformProjectsLocationsEndpointsDeployModelRequest object.

  Fields:
    endpoint: Required. The name of the Endpoint resource into which to deploy
      a Model. Format:
      `projects/{project}/locations/{location}/endpoints/{endpoint}`
    googleCloudAiplatformV1DeployModelRequest: A
      GoogleCloudAiplatformV1DeployModelRequest resource to be passed as the
      request body.
  """
    endpoint = _messages.StringField(1, required=True)
    googleCloudAiplatformV1DeployModelRequest = _messages.MessageField('GoogleCloudAiplatformV1DeployModelRequest', 2)