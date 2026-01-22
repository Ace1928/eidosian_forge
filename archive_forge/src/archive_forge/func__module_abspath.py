import gc
from io import BytesIO
import logging
import os
import os.path as osp
import stat
import uuid
import git
from git.cmd import Git
from git.compat import defenc
from git.config import GitConfigParser, SectionConstraint, cp
from git.exc import (
from git.objects.base import IndexObject, Object
from git.objects.util import TraversableIterableObj
from git.util import (
from .util import (
from typing import Callable, Dict, Mapping, Sequence, TYPE_CHECKING, cast
from typing import Any, Iterator, Union
from git.types import Commit_ish, Literal, PathLike, TBD
@classmethod
def _module_abspath(cls, parent_repo: 'Repo', path: PathLike, name: str) -> PathLike:
    if cls._need_gitfile_submodules(parent_repo.git):
        return osp.join(parent_repo.git_dir, 'modules', name)
    if parent_repo.working_tree_dir:
        return osp.join(parent_repo.working_tree_dir, path)
    raise NotADirectoryError()