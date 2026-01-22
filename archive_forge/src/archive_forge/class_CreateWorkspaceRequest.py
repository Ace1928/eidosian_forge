from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class CreateWorkspaceRequest(_messages.Message):
    """Request for CreateWorkspace.

  Fields:
    actions: An ordered sequence of actions to perform in the workspace. Can
      be empty. Specifying actions here instead of using ModifyWorkspace saves
      one RPC.
    repoId: The repo within which to create the workspace.
    workspace: The following fields of workspace, with the allowable exception
      of baseline, must be set. No other fields of workspace should be set.
      id.name Provides the name of the workspace and must be unique within the
      repo. Note: Do not set field id.repo_id.  The repo_id is provided above
      as a CreateWorkspaceRequest field.  alias: If alias names an existing
      movable alias, the workspace's baseline is set to the alias's revision.
      If alias does not name an existing movable alias, then the workspace is
      created with no baseline. When the workspace is committed, a new root
      revision is created with no parents. The new revision becomes the
      workspace's baseline and the alias name is used to create a movable
      alias referring to the revision.  baseline: A revision ID (hexadecimal
      string) for sequencing. If non-empty, alias must name an existing
      movable alias and baseline must match the alias's revision ID.
  """
    actions = _messages.MessageField('Action', 1, repeated=True)
    repoId = _messages.MessageField('RepoId', 2)
    workspace = _messages.MessageField('Workspace', 3)