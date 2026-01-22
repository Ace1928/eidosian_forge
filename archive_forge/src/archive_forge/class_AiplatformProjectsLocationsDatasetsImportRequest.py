from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AiplatformProjectsLocationsDatasetsImportRequest(_messages.Message):
    """A AiplatformProjectsLocationsDatasetsImportRequest object.

  Fields:
    googleCloudAiplatformV1ImportDataRequest: A
      GoogleCloudAiplatformV1ImportDataRequest resource to be passed as the
      request body.
    name: Required. The name of the Dataset resource. Format:
      `projects/{project}/locations/{location}/datasets/{dataset}`
  """
    googleCloudAiplatformV1ImportDataRequest = _messages.MessageField('GoogleCloudAiplatformV1ImportDataRequest', 1)
    name = _messages.StringField(2, required=True)