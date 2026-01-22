from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class ListWorkspacesResponse(_messages.Message):
    """Response for ListWorkspaces.

  Fields:
    workspaces: The listed workspaces.
  """
    workspaces = _messages.MessageField('Workspace', 1, repeated=True)