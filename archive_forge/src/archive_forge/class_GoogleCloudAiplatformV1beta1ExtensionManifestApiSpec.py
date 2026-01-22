from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1ExtensionManifestApiSpec(_messages.Message):
    """The API specification shown to the LLM.

  Fields:
    openApiGcsUri: Cloud Storage URI pointing to the OpenAPI spec.
    openApiYaml: The API spec in Open API standard and YAML format.
  """
    openApiGcsUri = _messages.StringField(1)
    openApiYaml = _messages.StringField(2)