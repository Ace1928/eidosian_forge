from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class ResolveFilesRequest(_messages.Message):
    """Request for ResolveFiles.

  Fields:
    resolvedPaths: Files that should be marked as resolved in the workspace.
      All files in resolved_paths must currently be in the CONFLICTED state in
      Workspace.changed_files.  NOTE: Changing a file's contents to match the
      contents in the workspace baseline, then calling ResolveFiles on it,
      will cause the file to be removed from the changed_files list entirely.
      If resolved_paths is empty, INVALID_ARGUMENT is returned. If
      resolved_paths contains duplicates, INVALID_ARGUMENT is returned. If
      resolved_paths contains a path that was never unresolved, or has already
      been resolved, FAILED_PRECONDITION is returned.
    workspaceId: The ID of the workspace.
  """
    resolvedPaths = _messages.StringField(1, repeated=True)
    workspaceId = _messages.MessageField('CloudWorkspaceId', 2)