import os
from ... import config as _mod_config
from ... import osutils, ui
from ...bzr.generate_ids import gen_revision_id
from ...bzr.inventorytree import InventoryTreeChange
from ...errors import (BzrError, NoCommonAncestor, UnknownFormatError,
from ...graph import FrozenHeadsCache
from ...merge import Merger
from ...revision import NULL_REVISION
from ...trace import mutter
from ...transport import NoSuchFile
from ...tsort import topo_sort
from .maptree import MapTree, map_file_ids
def wrap_iter_changes(old_iter_changes, map_tree):
    for change in old_iter_changes:
        if change.parent_id[0] is not None:
            old_parent = map_tree.new_id(change.parent_id[0])
        else:
            old_parent = change.parent_id[0]
        if change.parent_id[1] is not None:
            new_parent = map_tree.new_id(change.parent_id[1])
        else:
            new_parent = change.parent_id[1]
        yield InventoryTreeChange(map_tree.new_id(change.file_id), change.path, change.changed_content, change.versioned, (old_parent, new_parent), change.name, change.kind, change.executable)