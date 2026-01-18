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
def unmerged_blobs(self) -> Dict[PathLike, List[Tuple[StageType, Blob]]]:
    """
        :return:
            Dict(path : list( tuple( stage, Blob, ...))), being
            a dictionary associating a path in the index with a list containing
            sorted stage/blob pairs.

        :note:
            Blobs that have been removed in one side simply do not exist in the
            given stage. I.e. a file removed on the 'other' branch whose entries
            are at stage 3 will not have a stage 3 entry.
        """
    is_unmerged_blob = lambda t: t[0] != 0
    path_map: Dict[PathLike, List[Tuple[StageType, Blob]]] = {}
    for stage, blob in self.iter_blobs(is_unmerged_blob):
        path_map.setdefault(blob.path, []).append((stage, blob))
    for line in path_map.values():
        line.sort()
    return path_map