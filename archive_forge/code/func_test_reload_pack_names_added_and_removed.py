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
def test_reload_pack_names_added_and_removed(self):
    tree, r, packs, revs = self.make_packs_and_alt_repo()
    names = packs.names()
    tree.branch.repository.pack()
    new_names = tree.branch.repository._pack_collection.names()
    self.assertEqual(names, packs.names())
    self.assertTrue(packs.reload_pack_names())
    self.assertEqual(new_names, packs.names())
    self.assertEqual({revs[-1]: (revs[-2],)}, r.get_parent_map([revs[-1]]))
    self.assertFalse(packs.reload_pack_names())