from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AiplatformProjectsLocationsExtensionsExecuteRequest(_messages.Message):
    """A AiplatformProjectsLocationsExtensionsExecuteRequest object.

  Fields:
    googleCloudAiplatformV1beta1ExecuteExtensionRequest: A
      GoogleCloudAiplatformV1beta1ExecuteExtensionRequest resource to be
      passed as the request body.
    name: Required. Name (identifier) of the extension; Format:
      `projects/{project}/locations/{location}/extensions/{extension}`
  """
    googleCloudAiplatformV1beta1ExecuteExtensionRequest = _messages.MessageField('GoogleCloudAiplatformV1beta1ExecuteExtensionRequest', 1)
    name = _messages.StringField(2, required=True)