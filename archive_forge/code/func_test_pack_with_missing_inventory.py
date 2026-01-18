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
def test_pack_with_missing_inventory(self):
    repo, rev_a_pack_name, inv_a_pack_name, rev_c_pack_name = self.make_branch_with_disjoint_inventory_and_revision()
    inv_a_pack = repo._pack_collection.get_pack_by_name(inv_a_pack_name)
    repo._pack_collection._remove_pack_from_memory(inv_a_pack)
    packer = groupcompress_repo.GCCHKPacker(repo._pack_collection, repo._pack_collection.all_packs(), '.test-pack')
    e = self.assertRaises(ValueError, packer.pack)
    packer.new_pack.abort()
    self.assertContainsRe(str(e), "We are missing inventories for revisions: .*'A'")