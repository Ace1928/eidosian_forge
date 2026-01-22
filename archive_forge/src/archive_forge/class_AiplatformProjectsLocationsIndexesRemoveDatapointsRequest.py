from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AiplatformProjectsLocationsIndexesRemoveDatapointsRequest(_messages.Message):
    """A AiplatformProjectsLocationsIndexesRemoveDatapointsRequest object.

  Fields:
    googleCloudAiplatformV1RemoveDatapointsRequest: A
      GoogleCloudAiplatformV1RemoveDatapointsRequest resource to be passed as
      the request body.
    index: Required. The name of the Index resource to be updated. Format:
      `projects/{project}/locations/{location}/indexes/{index}`
  """
    googleCloudAiplatformV1RemoveDatapointsRequest = _messages.MessageField('GoogleCloudAiplatformV1RemoveDatapointsRequest', 1)
    index = _messages.StringField(2, required=True)