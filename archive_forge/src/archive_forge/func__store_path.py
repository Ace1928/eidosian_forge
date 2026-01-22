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
def _store_path(self, filepath: PathLike, fprogress: Callable) -> BaseIndexEntry:
    """Store file at filepath in the database and return the base index entry
        Needs the git_working_dir decorator active ! This must be assured in the calling code"""
    st = os.lstat(filepath)
    if S_ISLNK(st.st_mode):
        open_stream: Callable[[], BinaryIO] = lambda: BytesIO(force_bytes(os.readlink(filepath), encoding=defenc))
    else:
        open_stream = lambda: open(filepath, 'rb')
    with open_stream() as stream:
        fprogress(filepath, False, filepath)
        istream = self.repo.odb.store(IStream(Blob.type, st.st_size, stream))
        fprogress(filepath, True, filepath)
    return BaseIndexEntry((stat_mode_to_index_mode(st.st_mode), istream.binsha, 0, to_native_path_linux(filepath)))