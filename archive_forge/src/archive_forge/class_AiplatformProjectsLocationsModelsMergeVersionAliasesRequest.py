from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AiplatformProjectsLocationsModelsMergeVersionAliasesRequest(_messages.Message):
    """A AiplatformProjectsLocationsModelsMergeVersionAliasesRequest object.

  Fields:
    googleCloudAiplatformV1MergeVersionAliasesRequest: A
      GoogleCloudAiplatformV1MergeVersionAliasesRequest resource to be passed
      as the request body.
    name: Required. The name of the model version to merge aliases, with a
      version ID explicitly included. Example:
      `projects/{project}/locations/{location}/models/{model}@1234`
  """
    googleCloudAiplatformV1MergeVersionAliasesRequest = _messages.MessageField('GoogleCloudAiplatformV1MergeVersionAliasesRequest', 1)
    name = _messages.StringField(2, required=True)