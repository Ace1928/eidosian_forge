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
def _write_path_to_stdin(self, proc: 'Popen', filepath: PathLike, item: PathLike, fmakeexc: Callable[..., GitError], fprogress: Callable[[PathLike, bool, PathLike], None], read_from_stdout: bool=True) -> Union[None, str]:
    """Write path to proc.stdin and make sure it processes the item, including progress.

        :return: stdout string

        :param read_from_stdout: if True, proc.stdout will be read after the item
            was sent to stdin. In that case, it will return None.

        :note: There is a bug in git-update-index that prevents it from sending
            reports just in time. This is why we have a version that tries to
            read stdout and one which doesn't. In fact, the stdout is not
            important as the piped-in files are processed anyway and just in time.

        :note: Newlines are essential here, gits behaviour is somewhat inconsistent
            on this depending on the version, hence we try our best to deal with
            newlines carefully. Usually the last newline will not be sent, instead
            we will close stdin to break the pipe.
        """
    fprogress(filepath, False, item)
    rval: Union[None, str] = None
    if proc.stdin is not None:
        try:
            proc.stdin.write(('%s\n' % filepath).encode(defenc))
        except IOError as e:
            raise fmakeexc() from e
        proc.stdin.flush()
    if read_from_stdout and proc.stdout is not None:
        rval = proc.stdout.readline().strip()
    fprogress(filepath, True, item)
    return rval