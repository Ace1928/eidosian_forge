import os
import stat
import sys
import warnings
from contextlib import suppress
from io import BytesIO
from typing import (
from .errors import NotTreeError
from .file import GitFile
from .objects import (
from .pack import (
from .protocol import DEPTH_INFINITE
from .refs import PEELED_TAG_SUFFIX, Ref
def _collect_filetree_revs(obj_store: ObjectContainer, tree_sha: ObjectID, kset: Set[ObjectID]) -> None:
    """Collect SHA1s of files and directories for specified tree.

    Args:
      obj_store: Object store to get objects by SHA from
      tree_sha: tree reference to walk
      kset: set to fill with references to files and directories
    """
    filetree = obj_store[tree_sha]
    assert isinstance(filetree, Tree)
    for name, mode, sha in filetree.iteritems():
        if not S_ISGITLINK(mode) and sha not in kset:
            kset.add(sha)
            if stat.S_ISDIR(mode):
                _collect_filetree_revs(obj_store, sha, kset)