import stat
from typing import Dict, Tuple
from fastimport import commands, parser, processor
from fastimport import errors as fastimport_errors
from .index import commit_tree
from .object_store import iter_tree_contents
from .objects import ZERO_SHA, Blob, Commit, Tag
def _export_commit(self, commit, ref, base_tree=None):
    file_cmds = list(self._iter_files(base_tree, commit.tree))
    marker = self._allocate_marker()
    if commit.parents:
        from_ = commit.parents[0]
        merges = commit.parents[1:]
    else:
        from_ = None
        merges = []
    author, author_email = split_email(commit.author)
    committer, committer_email = split_email(commit.committer)
    cmd = commands.CommitCommand(ref, marker, (author, author_email, commit.author_time, commit.author_timezone), (committer, committer_email, commit.commit_time, commit.commit_timezone), commit.message, from_, merges, file_cmds)
    return (cmd, marker)