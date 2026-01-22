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
def _preprocess_add_items(self, items: Sequence[Union[PathLike, Blob, BaseIndexEntry, 'Submodule']]) -> Tuple[List[PathLike], List[BaseIndexEntry]]:
    """Split the items into two lists of path strings and BaseEntries."""
    paths = []
    entries = []
    if isinstance(items, (str, os.PathLike)):
        items = [items]
    for item in items:
        if isinstance(item, (str, os.PathLike)):
            paths.append(self._to_relative_path(item))
        elif isinstance(item, (Blob, Submodule)):
            entries.append(BaseIndexEntry.from_blob(item))
        elif isinstance(item, BaseIndexEntry):
            entries.append(item)
        else:
            raise TypeError('Invalid Type: %r' % item)
    return (paths, entries)