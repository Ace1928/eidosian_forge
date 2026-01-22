import errno
import os
import posixpath
import stat
from collections import deque
from functools import partial
from io import BytesIO
from typing import Union, List, Tuple, Set
from dulwich.config import ConfigFile as GitConfigFile
from dulwich.config import parse_submodules
from dulwich.diff_tree import RenameDetector, tree_changes
from dulwich.errors import NotTreeError
from dulwich.index import (Index, IndexEntry, blob_from_path_and_stat,
from dulwich.object_store import OverlayObjectStore, iter_tree_contents, BaseObjectStore
from dulwich.objects import S_IFGITLINK, S_ISGITLINK, ZERO_SHA, Blob, Tree, ObjectID
from .. import controldir as _mod_controldir
from .. import delta, errors, mutabletree, osutils, revisiontree, trace
from .. import transport as _mod_transport
from .. import tree as _mod_tree
from .. import urlutils, workingtree
from ..bzr.inventorytree import InventoryTreeChange
from ..revision import CURRENT_REVISION, NULL_REVISION
from ..transport import get_transport
from ..tree import MissingNestedTree, TreeEntry
from .mapping import (decode_git_path, default_mapping, encode_git_path,
def _lookup_path(self, path):
    if self.tree is None:
        raise _mod_transport.NoSuchFile(path)
    encoded_path = encode_git_path(path)
    parts = encoded_path.split(b'/')
    hexsha = self.tree
    store = self.store
    mode = None
    for i, p in enumerate(parts):
        if not p:
            continue
        obj = store[hexsha]
        if not isinstance(obj, Tree):
            raise NotTreeError(hexsha)
        try:
            mode, hexsha = obj[p]
        except KeyError:
            raise _mod_transport.NoSuchFile(path)
        if S_ISGITLINK(mode) and i != len(parts) - 1:
            store = self._get_submodule_store(b'/'.join(parts[:i + 1]))
            hexsha = store[hexsha].tree
    return (store, mode, hexsha)