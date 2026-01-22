from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DialogflowProjectsAgentVersionsPatchRequest(_messages.Message):
    """A DialogflowProjectsAgentVersionsPatchRequest object.

  Fields:
    googleCloudDialogflowV2Version: A GoogleCloudDialogflowV2Version resource
      to be passed as the request body.
    name: Output only. The unique identifier of this agent version. Supported
      formats: - `projects//agent/versions/` -
      `projects//locations//agent/versions/`
    updateMask: Required. The mask to control which fields get updated.
  """
    googleCloudDialogflowV2Version = _messages.MessageField('GoogleCloudDialogflowV2Version', 1)
    name = _messages.StringField(2, required=True)
    updateMask = _messages.StringField(3)