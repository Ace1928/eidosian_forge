from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AiplatformProjectsLocationsNotebookRuntimesGenerateAccessTokenRequest(_messages.Message):
    """A AiplatformProjectsLocationsNotebookRuntimesGenerateAccessTokenRequest
  object.

  Fields:
    googleCloudAiplatformV1beta1GenerateAccessTokenRequest: A
      GoogleCloudAiplatformV1beta1GenerateAccessTokenRequest resource to be
      passed as the request body.
    name: Required. The name of the resource requesting the OAuth2 token.
      Format: `projects/{project}/locations/{location}/notebookRuntimes/{noteb
      ook_runtime}` `projects/{project}/locations/{location}/notebookExecution
      Jobs/{notebook_execution_job}`
  """
    googleCloudAiplatformV1beta1GenerateAccessTokenRequest = _messages.MessageField('GoogleCloudAiplatformV1beta1GenerateAccessTokenRequest', 1)
    name = _messages.StringField(2, required=True)