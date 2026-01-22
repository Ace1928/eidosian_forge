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
def _path2ie(self, path):
    """Lookup an inventory entry by path.

        :param path: Path to look up
        :return: InventoryEntry
        """
    inv, ie = self._path2inv_ie(path)
    if ie is None:
        raise _mod_transport.NoSuchFile(path)
    return ie