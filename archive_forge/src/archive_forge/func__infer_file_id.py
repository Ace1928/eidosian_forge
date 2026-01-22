import errno
import os
import re
from ..lazy_import import lazy_import
from breezy import (
from .. import transport as _mod_transport
from ..conflicts import Conflict as BaseConflict
from ..conflicts import ConflictList as BaseConflictList
from . import rio
def _infer_file_id(self, tree):
    possible_paths = []
    for p in (self.path, self.conflict_path):
        if p == '<deleted>':
            continue
        if p is not None:
            possible_paths.append(p)
    file_id = None
    for revid in tree.get_parent_ids():
        revtree = self._revision_tree(tree, revid)
        for p in possible_paths:
            file_id = revtree.path2id(p)
            if file_id is not None:
                return (revtree, file_id)
    return (None, None)