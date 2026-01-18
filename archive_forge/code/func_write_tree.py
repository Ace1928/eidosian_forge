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
def write_tree(self) -> Tree:
    """Write this index to a corresponding Tree object into the repository's
        object database and return it.

        :return: Tree object representing this index.

        :note: The tree will be written even if one or more objects the tree refers to
            does not yet exist in the object database. This could happen if you added
            Entries to the index directly.

        :raise ValueError: if there are no entries in the cache

        :raise UnmergedEntriesError:
        """
    mdb = MemoryDB()
    entries = self._entries_sorted()
    binsha, tree_items = write_tree_from_cache(entries, mdb, slice(0, len(entries)))
    mdb.stream_copy(mdb.sha_iter(), self.repo.odb)
    root_tree = Tree(self.repo, binsha, path='')
    root_tree._cache = tree_items
    return root_tree