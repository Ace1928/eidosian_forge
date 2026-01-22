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
def _set_merges_from_parent_ids(self, rhs_parent_ids):
    try:
        merges = [self.branch.lookup_bzr_revision_id(revid)[0] for revid in rhs_parent_ids]
    except errors.NoSuchRevision as e:
        raise errors.GhostRevisionUnusableHere(e.revision)
    if merges:
        self.control_transport.put_bytes('MERGE_HEAD', b'\n'.join(merges), mode=self.controldir._get_file_mode())
    else:
        try:
            self.control_transport.delete('MERGE_HEAD')
        except _mod_transport.NoSuchFile:
            pass