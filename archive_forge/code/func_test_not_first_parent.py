from stat import S_ISDIR
import breezy
from breezy import controldir, errors, osutils, repository
from breezy import revision as _mod_revision
from breezy import tests, transport, upgrade, workingtree
from breezy.bzr import (btree_index, bzrdir, groupcompress_repo, inventory,
from breezy.bzr import repository as bzrrepository
from breezy.bzr import versionedfile, vf_repository, vf_search
from breezy.bzr.btree_index import BTreeBuilder, BTreeGraphIndex
from breezy.bzr.index import GraphIndex
from breezy.errors import UnknownFormatError
from breezy.repository import RepositoryFormat
from breezy.tests import TestCase, TestCaseWithTransport
def test_not_first_parent(self):
    self.builder.build_snapshot(None, [], revision_id=b'revid1')
    self.builder.build_snapshot([b'revid1'], [], revision_id=b'revid2')
    self.builder.build_snapshot([b'revid2'], [], revision_id=b'revid3')
    rev_set = [b'revid3', b'revid2']
    self.assertParentIds([b'revid1'], rev_set)