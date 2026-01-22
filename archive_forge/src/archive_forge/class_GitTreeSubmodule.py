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
class GitTreeSubmodule(_mod_tree.TreeReference):
    __slots__ = ['file_id', 'name', 'parent_id', 'reference_revision', 'git_sha1']

    def __init__(self, file_id, name, parent_id, reference_revision=None, git_sha1=None):
        self.file_id = file_id
        self.name = name
        self.parent_id = parent_id
        self.reference_revision = reference_revision
        self.git_sha1 = git_sha1

    @property
    def executable(self):
        return False

    @property
    def kind(self):
        return 'tree-reference'

    def __repr__(self):
        return '%s(file_id=%r, name=%r, parent_id=%r, reference_revision=%r)' % (type(self).__name__, self.file_id, self.name, self.parent_id, self.reference_revision)

    def __eq__(self, other):
        return self.kind == other.kind and self.file_id == other.file_id and (self.name == other.name) and (self.parent_id == other.parent_id) and (self.reference_revision == other.reference_revision)

    def copy(self):
        return self.__class__(self.file_id, self.name, self.parent_id, self.reference_revision)