from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class RefreshWorkspaceRequest(_messages.Message):
    """Request for RefreshWorkspace.

  Fields:
    workspaceId: The ID of the workspace.
  """
    workspaceId = _messages.MessageField('CloudWorkspaceId', 1)