from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AiplatformProjectsLocationsMetadataStoresExecutionsPurgeRequest(_messages.Message):
    """A AiplatformProjectsLocationsMetadataStoresExecutionsPurgeRequest
  object.

  Fields:
    googleCloudAiplatformV1PurgeExecutionsRequest: A
      GoogleCloudAiplatformV1PurgeExecutionsRequest resource to be passed as
      the request body.
    parent: Required. The metadata store to purge Executions from. Format:
      `projects/{project}/locations/{location}/metadataStores/{metadatastore}`
  """
    googleCloudAiplatformV1PurgeExecutionsRequest = _messages.MessageField('GoogleCloudAiplatformV1PurgeExecutionsRequest', 1)
    parent = _messages.StringField(2, required=True)