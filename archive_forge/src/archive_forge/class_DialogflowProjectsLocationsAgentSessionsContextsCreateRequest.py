from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DialogflowProjectsLocationsAgentSessionsContextsCreateRequest(_messages.Message):
    """A DialogflowProjectsLocationsAgentSessionsContextsCreateRequest object.

  Fields:
    googleCloudDialogflowV2Context: A GoogleCloudDialogflowV2Context resource
      to be passed as the request body.
    parent: Required. The session to create a context for. Format:
      `projects//agent/sessions/` or
      `projects//agent/environments//users//sessions/`. If `Environment ID` is
      not specified, we assume default 'draft' environment. If `User ID` is
      not specified, we assume default '-' user.
  """
    googleCloudDialogflowV2Context = _messages.MessageField('GoogleCloudDialogflowV2Context', 1)
    parent = _messages.StringField(2, required=True)