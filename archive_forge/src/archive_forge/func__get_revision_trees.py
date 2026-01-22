import re
import sys
import time
from email.utils import parseaddr
import breezy.branch
import breezy.revision
from ... import (builtins, errors, lazy_import, lru_cache, osutils, progress,
from ... import transport as _mod_transport
from . import helpers, marks_file
from fastimport import commands
def _get_revision_trees(self, revids):
    missing = []
    by_revid = {}
    for revid in revids:
        if revid == breezy.revision.NULL_REVISION:
            by_revid[revid] = self.branch.repository.revision_tree(revid)
        elif revid not in self.tree_cache:
            missing.append(revid)
    for tree in self.branch.repository.revision_trees(missing):
        by_revid[tree.get_revision_id()] = tree
    for revid in revids:
        try:
            yield self.tree_cache[revid]
        except KeyError:
            yield by_revid[revid]
    for revid, tree in by_revid.items():
        self.tree_cache[revid] = tree