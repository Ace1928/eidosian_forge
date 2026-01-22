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
class GitTreeFile(_mod_tree.TreeFile):
    __slots__ = ['file_id', 'name', 'parent_id', 'text_size', 'executable', 'git_sha1']

    def __init__(self, file_id, name, parent_id, text_size=None, git_sha1=None, executable=None):
        self.file_id = file_id
        self.name = name
        self.parent_id = parent_id
        self.text_size = text_size
        self.git_sha1 = git_sha1
        self.executable = executable

    @property
    def kind(self):
        return 'file'

    def __eq__(self, other):
        return self.kind == other.kind and self.file_id == other.file_id and (self.name == other.name) and (self.parent_id == other.parent_id) and (self.git_sha1 == other.git_sha1) and (self.text_size == other.text_size) and (self.executable == other.executable)

    def __repr__(self):
        return '%s(file_id=%r, name=%r, parent_id=%r, text_size=%r, git_sha1=%r, executable=%r)' % (type(self).__name__, self.file_id, self.name, self.parent_id, self.text_size, self.git_sha1, self.executable)

    def copy(self):
        ret = self.__class__(self.file_id, self.name, self.parent_id)
        ret.git_sha1 = self.git_sha1
        ret.text_size = self.text_size
        ret.executable = self.executable
        return ret