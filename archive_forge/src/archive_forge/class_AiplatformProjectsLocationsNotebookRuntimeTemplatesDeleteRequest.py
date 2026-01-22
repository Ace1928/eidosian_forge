from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AiplatformProjectsLocationsNotebookRuntimeTemplatesDeleteRequest(_messages.Message):
    """A AiplatformProjectsLocationsNotebookRuntimeTemplatesDeleteRequest
  object.

  Fields:
    name: Required. The name of the NotebookRuntimeTemplate resource to be
      deleted. Format: `projects/{project}/locations/{location}/notebookRuntim
      eTemplates/{notebook_runtime_template}`
  """
    name = _messages.StringField(1, required=True)