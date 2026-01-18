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
def to_full_path(cls, path: Union[PathLike, 'SymbolicReference']) -> PathLike:
    """
        :return: string with a full repository-relative path which can be used to initialize
            a Reference instance, for instance by using
            :meth:`Reference.from_path <git.refs.reference.Reference.from_path>`.
        """
    if isinstance(path, SymbolicReference):
        path = path.path
    full_ref_path = path
    if not cls._common_path_default:
        return full_ref_path
    if not str(path).startswith(cls._common_path_default + '/'):
        full_ref_path = '%s/%s' % (cls._common_path_default, path)
    return full_ref_path