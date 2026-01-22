from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DialogflowProjectsLocationsAgentRestoreRequest(_messages.Message):
    """A DialogflowProjectsLocationsAgentRestoreRequest object.

  Fields:
    googleCloudDialogflowV2RestoreAgentRequest: A
      GoogleCloudDialogflowV2RestoreAgentRequest resource to be passed as the
      request body.
    parent: Required. The project that the agent to restore is associated
      with. Format: `projects/`.
  """
    googleCloudDialogflowV2RestoreAgentRequest = _messages.MessageField('GoogleCloudDialogflowV2RestoreAgentRequest', 1)
    parent = _messages.StringField(2, required=True)