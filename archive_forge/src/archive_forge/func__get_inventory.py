import base64
import os
import pprint
from io import BytesIO
from ... import cache_utf8, osutils, timestamp
from ...errors import BzrError, NoSuchId, TestamentMismatch
from ...osutils import pathjoin, sha_string, sha_strings
from ...revision import NULL_REVISION, Revision
from ...trace import mutter, warning
from ...tree import InterTree, Tree
from ..inventory import (Inventory, InventoryDirectory, InventoryFile,
from ..inventorytree import InventoryTree
from ..testament import StrictTestament
from ..xml5 import serializer_v5
from . import apply_bundle
def _get_inventory(self):
    """Build up the inventory entry for the BundleTree.

        This need to be called before ever accessing self.inventory
        """
    from os.path import basename, dirname
    inv = Inventory(None, self.revision_id)

    def add_entry(path, file_id):
        if path == '':
            parent_id = None
        else:
            parent_path = dirname(path)
            parent_id = self.path2id(parent_path)
        kind = self.kind(path)
        revision_id = self.get_last_changed(path)
        name = basename(path)
        if kind == 'directory':
            ie = InventoryDirectory(file_id, name, parent_id)
        elif kind == 'file':
            ie = InventoryFile(file_id, name, parent_id)
            ie.executable = self.is_executable(path)
        elif kind == 'symlink':
            ie = InventoryLink(file_id, name, parent_id)
            ie.symlink_target = self.get_symlink_target(path)
        ie.revision = revision_id
        if kind == 'file':
            ie.text_size, ie.text_sha1 = self.get_size_and_sha1(path)
            if ie.text_size is None:
                raise BzrError('Got a text_size of None for file_id %r' % file_id)
        inv.add(ie)
    sorted_entries = self.sorted_path_id()
    for path, file_id in sorted_entries:
        add_entry(path, file_id)
    return inv