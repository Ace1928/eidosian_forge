from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AiplatformProjectsLocationsNotebookRuntimesStartRequest(_messages.Message):
    """A AiplatformProjectsLocationsNotebookRuntimesStartRequest object.

  Fields:
    googleCloudAiplatformV1StartNotebookRuntimeRequest: A
      GoogleCloudAiplatformV1StartNotebookRuntimeRequest resource to be passed
      as the request body.
    name: Required. The name of the NotebookRuntime resource to be started.
      Instead of checking whether the name is in valid NotebookRuntime
      resource name format, directly throw NotFound exception if there is no
      such NotebookRuntime in spanner.
  """
    googleCloudAiplatformV1StartNotebookRuntimeRequest = _messages.MessageField('GoogleCloudAiplatformV1StartNotebookRuntimeRequest', 1)
    name = _messages.StringField(2, required=True)