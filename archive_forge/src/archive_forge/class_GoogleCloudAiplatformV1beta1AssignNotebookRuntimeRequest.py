from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1AssignNotebookRuntimeRequest(_messages.Message):
    """Request message for NotebookService.AssignNotebookRuntime.

  Fields:
    notebookRuntime: Required. Provide runtime specific information (e.g.
      runtime owner, notebook id) used for NotebookRuntime assignment.
    notebookRuntimeId: Optional. User specified ID for the notebook runtime.
    notebookRuntimeTemplate: Required. The resource name of the
      NotebookRuntimeTemplate based on which a NotebookRuntime will be
      assigned (reuse or create a new one).
  """
    notebookRuntime = _messages.MessageField('GoogleCloudAiplatformV1beta1NotebookRuntime', 1)
    notebookRuntimeId = _messages.StringField(2)
    notebookRuntimeTemplate = _messages.StringField(3)