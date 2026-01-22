import os
from git.compat import defenc
from git.objects import Object
from git.objects.commit import Commit
from git.util import (
from gitdb.exc import BadObject, BadName
from .log import RefLog
from typing import (
from git.types import Commit_ish, PathLike
def _get_reference(self) -> 'SymbolicReference':
    """
        :return: Reference Object we point to

        :raise TypeError: If this symbolic reference is detached, hence it doesn't point
            to a reference, but to a commit"""
    sha, target_ref_path = self._get_ref_info(self.repo, self.path)
    if target_ref_path is None:
        raise TypeError('%s is a detached symbolic reference as it points to %r' % (self, sha))
    return self.from_path(self.repo, target_ref_path)