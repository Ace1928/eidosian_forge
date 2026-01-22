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
def _path2inv_ie(self, path):
    inv = self.root_inventory
    if isinstance(path, list):
        remaining = path
    else:
        remaining = osutils.splitpath(path)
    ie = inv.root
    while remaining:
        ie, base, remaining = inv.get_entry_by_path_partial(remaining)
        if remaining:
            inv = self._get_nested_tree('/'.join(base), ie.file_id, ie.reference_revision).root_inventory
    if ie is None:
        return (None, None)
    return (inv, ie)