from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AiplatformProjectsLocationsNotebookRuntimesUpgradeRequest(_messages.Message):
    """A AiplatformProjectsLocationsNotebookRuntimesUpgradeRequest object.

  Fields:
    googleCloudAiplatformV1beta1UpgradeNotebookRuntimeRequest: A
      GoogleCloudAiplatformV1beta1UpgradeNotebookRuntimeRequest resource to be
      passed as the request body.
    name: Required. The name of the NotebookRuntime resource to be upgrade.
      Instead of checking whether the name is in valid NotebookRuntime
      resource name format, directly throw NotFound exception if there is no
      such NotebookRuntime in spanner.
  """
    googleCloudAiplatformV1beta1UpgradeNotebookRuntimeRequest = _messages.MessageField('GoogleCloudAiplatformV1beta1UpgradeNotebookRuntimeRequest', 1)
    name = _messages.StringField(2, required=True)