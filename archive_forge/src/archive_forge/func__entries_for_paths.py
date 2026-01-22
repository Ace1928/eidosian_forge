import contextlib
import datetime
import glob
from io import BytesIO
import os
from stat import S_ISLNK
import subprocess
import tempfile
from git.compat import (
from git.exc import GitCommandError, CheckoutError, GitError, InvalidGitRepositoryError
from git.objects import (
from git.objects.util import Serializable
from git.util import (
from gitdb.base import IStream
from gitdb.db import MemoryDB
import git.diff as git_diff
import os.path as osp
from .fun import (
from .typ import (
from .util import TemporaryFileSwap, post_clear_cache, default_index, git_working_dir
from typing import (
from git.types import Commit_ish, PathLike
@unbare_repo
@git_working_dir
def _entries_for_paths(self, paths: List[str], path_rewriter: Union[Callable, None], fprogress: Callable, entries: List[BaseIndexEntry]) -> List[BaseIndexEntry]:
    entries_added: List[BaseIndexEntry] = []
    if path_rewriter:
        for path in paths:
            if osp.isabs(path):
                abspath = path
                gitrelative_path = path[len(str(self.repo.working_tree_dir)) + 1:]
            else:
                gitrelative_path = path
                if self.repo.working_tree_dir:
                    abspath = osp.join(self.repo.working_tree_dir, gitrelative_path)
            blob = Blob(self.repo, Blob.NULL_BIN_SHA, stat_mode_to_index_mode(os.stat(abspath).st_mode), to_native_path_linux(gitrelative_path))
            entries.append(BaseIndexEntry.from_blob(blob))
        del paths[:]
    assert len(entries_added) == 0
    for filepath in self._iter_expand_paths(paths):
        entries_added.append(self._store_path(filepath, fprogress))
    return entries_added