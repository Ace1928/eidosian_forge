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
def _get_ie(self, inv_path):
    """Retrieve the most up to date inventory entry for a path.

        :param inv_path: Normalized inventory path
        :return: Inventory entry (with possibly invalid .children for
            directories)
        """
    entry = self._invdelta.get(inv_path)
    if entry is not None:
        return entry[3]
    inv_path = self.tree._fix_case_of_inventory_path(inv_path)
    try:
        return next(self.tree.iter_entries_by_dir(specific_files=[inv_path]))[1]
    except StopIteration:
        return None