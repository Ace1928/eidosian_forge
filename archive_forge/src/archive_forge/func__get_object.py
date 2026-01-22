import os
from git.compat import defenc
from git.objects import Object
from git.objects.commit import Commit
from git.util import (
from gitdb.exc import BadObject, BadName
from .log import RefLog
from typing import (
from git.types import Commit_ish, PathLike
def _get_object(self) -> Commit_ish:
    """
        :return:
            The object our ref currently refers to. Refs can be cached, they will
            always point to the actual object as it gets re-created on each query.
        """
    return Object.new_from_sha(self.repo, hex_to_bin(self.dereference_recursive(self.repo, self.path)))