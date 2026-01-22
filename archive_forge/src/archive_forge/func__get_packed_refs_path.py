import os
from git.compat import defenc
from git.objects import Object
from git.objects.commit import Commit
from git.util import (
from gitdb.exc import BadObject, BadName
from .log import RefLog
from typing import (
from git.types import Commit_ish, PathLike
@classmethod
def _get_packed_refs_path(cls, repo: 'Repo') -> str:
    return os.path.join(repo.common_dir, 'packed-refs')