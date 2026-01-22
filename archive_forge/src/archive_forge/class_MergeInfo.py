from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class MergeInfo(_messages.Message):
    """MergeInfo holds information needed while resolving merges, and refreshes
  that involve conflicts.

  Fields:
    commonAncestorRevisionId: Revision ID of the closest common ancestor of
      the file trees that are participating in a refresh or merge.  During a
      refresh, the common ancestor is the baseline of the workspace.  During a
      merge of two branches, the common ancestor is derived from the workspace
      baseline and the alias of the branch being merged in.  The repository
      state at the common ancestor provides the base version for a three-way
      merge.
    isRefresh: If true, a refresh operation is in progress.  If false, a merge
      is in progress.
    otherRevisionId: During a refresh, the ID of the revision with which the
      workspace is being refreshed. This is the revision ID to which the
      workspace's alias refers at the time of the RefreshWorkspace call.
      During a merge, the ID of the revision that's being merged into the
      workspace's alias. This is the revision_id field of the MergeRequest.
    workspaceAfterSnapshotId: The workspace snapshot immediately after the
      refresh or merge RPC completes.  If a file has conflicts, this snapshot
      contains the version of the file with conflict markers.
    workspaceBeforeSnapshotId: During a refresh, the snapshot ID of the latest
      change to the workspace before the refresh.  During a merge, the
      workspace's baseline, which is identical to the commit hash of the
      workspace's alias before initiating the merge.
  """
    commonAncestorRevisionId = _messages.StringField(1)
    isRefresh = _messages.BooleanField(2)
    otherRevisionId = _messages.StringField(3)
    workspaceAfterSnapshotId = _messages.StringField(4)
    workspaceBeforeSnapshotId = _messages.StringField(5)