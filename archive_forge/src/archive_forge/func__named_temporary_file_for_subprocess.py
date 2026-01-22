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
@contextlib.contextmanager
def _named_temporary_file_for_subprocess(directory: PathLike) -> Generator[str, None, None]:
    """Create a named temporary file git subprocesses can open, deleting it afterward.

    :param directory: The directory in which the file is created.

    :return: A context manager object that creates the file and provides its name on
        entry, and deletes it on exit.
    """
    if os.name == 'nt':
        fd, name = tempfile.mkstemp(dir=directory)
        os.close(fd)
        try:
            yield name
        finally:
            os.remove(name)
    else:
        with tempfile.NamedTemporaryFile(dir=directory) as ctx:
            yield ctx.name