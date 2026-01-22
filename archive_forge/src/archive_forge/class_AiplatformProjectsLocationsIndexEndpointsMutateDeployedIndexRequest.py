from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AiplatformProjectsLocationsIndexEndpointsMutateDeployedIndexRequest(_messages.Message):
    """A AiplatformProjectsLocationsIndexEndpointsMutateDeployedIndexRequest
  object.

  Fields:
    googleCloudAiplatformV1DeployedIndex: A
      GoogleCloudAiplatformV1DeployedIndex resource to be passed as the
      request body.
    indexEndpoint: Required. The name of the IndexEndpoint resource into which
      to deploy an Index. Format: `projects/{project}/locations/{location}/ind
      exEndpoints/{index_endpoint}`
  """
    googleCloudAiplatformV1DeployedIndex = _messages.MessageField('GoogleCloudAiplatformV1DeployedIndex', 1)
    indexEndpoint = _messages.StringField(2, required=True)