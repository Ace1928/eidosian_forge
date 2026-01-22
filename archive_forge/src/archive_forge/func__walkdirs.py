import errno
import itertools
import os
import posixpath
import re
import stat
import sys
from collections import defaultdict
from dulwich.config import ConfigFile as GitConfigFile
from dulwich.file import FileLocked, GitFile
from dulwich.ignore import IgnoreFilterManager
from dulwich.index import (ConflictedIndexEntry, Index, IndexEntry, SHA1Writer,
from dulwich.object_store import iter_tree_contents
from dulwich.objects import S_ISGITLINK
from .. import branch as _mod_branch
from .. import conflicts as _mod_conflicts
from .. import controldir as _mod_controldir
from .. import errors, globbing, lock, osutils
from .. import revision as _mod_revision
from .. import trace
from .. import transport as _mod_transport
from .. import tree, urlutils, workingtree
from ..decorators import only_raises
from ..mutabletree import BadReferenceTarget, MutableTree
from .dir import BareLocalGitControlDirFormat, LocalGitDir
from .mapping import decode_git_path, encode_git_path, mode_kind
from .tree import MutableGitIndexTree
def _walkdirs(self, prefix=''):
    if prefix != '':
        prefix += '/'
    prefix = encode_git_path(prefix)
    per_dir = defaultdict(set)
    if prefix == b'':
        per_dir['', self.path2id('')] = set()

    def add_entry(path, kind):
        if path == b'' or not path.startswith(prefix):
            return
        dirname, child_name = posixpath.split(path)
        add_entry(dirname, 'directory')
        dirname = decode_git_path(dirname)
        dir_file_id = self.path2id(dirname)
        if not isinstance(value, (tuple, IndexEntry)):
            raise ValueError(value)
        per_dir[dirname, dir_file_id].add((decode_git_path(path), decode_git_path(child_name), kind, None, self.path2id(decode_git_path(path)), kind))
    with self.lock_read():
        for path, value in self.index.iteritems():
            if self.mapping.is_special_file(path):
                continue
            if not path.startswith(prefix):
                continue
            add_entry(path, mode_kind(value.mode))
    return ((k, sorted(v)) for k, v in sorted(per_dir.items()))