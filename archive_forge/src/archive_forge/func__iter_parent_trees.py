import os
import re
from collections import deque
from typing import TYPE_CHECKING, Optional, Type
from .. import branch as _mod_branch
from .. import controldir, debug, errors, lazy_import, osutils, revision, trace
from .. import transport as _mod_transport
from ..controldir import ControlDir
from ..mutabletree import MutableTree
from ..repository import Repository
from ..revisiontree import RevisionTree
from breezy import (
from breezy.bzr import (
from ..tree import (FileTimestampUnavailable, InterTree, MissingNestedTree,
def _iter_parent_trees(self):
    """Iterate through parent trees, defaulting to Tree.revision_tree."""
    for revision_id in self.get_parent_ids():
        try:
            yield self.revision_tree(revision_id)
        except errors.NoSuchRevisionInTree:
            yield self.branch.repository.revision_tree(revision_id)