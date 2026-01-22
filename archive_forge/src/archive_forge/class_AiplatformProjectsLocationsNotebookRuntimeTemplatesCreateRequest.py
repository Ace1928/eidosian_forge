from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AiplatformProjectsLocationsNotebookRuntimeTemplatesCreateRequest(_messages.Message):
    """A AiplatformProjectsLocationsNotebookRuntimeTemplatesCreateRequest
  object.

  Fields:
    googleCloudAiplatformV1NotebookRuntimeTemplate: A
      GoogleCloudAiplatformV1NotebookRuntimeTemplate resource to be passed as
      the request body.
    notebookRuntimeTemplateId: Optional. User specified ID for the notebook
      runtime template.
    parent: Required. The resource name of the Location to create the
      NotebookRuntimeTemplate. Format:
      `projects/{project}/locations/{location}`
  """
    googleCloudAiplatformV1NotebookRuntimeTemplate = _messages.MessageField('GoogleCloudAiplatformV1NotebookRuntimeTemplate', 1)
    notebookRuntimeTemplateId = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)