from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DialogflowProjectsAgentEnvironmentsUsersSessionsDeleteContextsRequest(_messages.Message):
    """A DialogflowProjectsAgentEnvironmentsUsersSessionsDeleteContextsRequest
  object.

  Fields:
    parent: Required. The name of the session to delete all contexts from.
      Format: `projects//agent/sessions/` or
      `projects//agent/environments//users//sessions/`. If `Environment ID` is
      not specified we assume default 'draft' environment. If `User ID` is not
      specified, we assume default '-' user.
  """
    parent = _messages.StringField(1, required=True)