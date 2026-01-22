from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AiplatformProjectsLocationsIndexEndpointsUndeployIndexRequest(_messages.Message):
    """A AiplatformProjectsLocationsIndexEndpointsUndeployIndexRequest object.

  Fields:
    googleCloudAiplatformV1UndeployIndexRequest: A
      GoogleCloudAiplatformV1UndeployIndexRequest resource to be passed as the
      request body.
    indexEndpoint: Required. The name of the IndexEndpoint resource from which
      to undeploy an Index. Format: `projects/{project}/locations/{location}/i
      ndexEndpoints/{index_endpoint}`
  """
    googleCloudAiplatformV1UndeployIndexRequest = _messages.MessageField('GoogleCloudAiplatformV1UndeployIndexRequest', 1)
    indexEndpoint = _messages.StringField(2, required=True)