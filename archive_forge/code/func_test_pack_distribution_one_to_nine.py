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
def test_pack_distribution_one_to_nine(self):
    packs = self.get_packs()
    self.assertEqual([1], packs.pack_distribution(1))
    self.assertEqual([1, 1], packs.pack_distribution(2))
    self.assertEqual([1, 1, 1], packs.pack_distribution(3))
    self.assertEqual([1, 1, 1, 1], packs.pack_distribution(4))
    self.assertEqual([1, 1, 1, 1, 1], packs.pack_distribution(5))
    self.assertEqual([1, 1, 1, 1, 1, 1], packs.pack_distribution(6))
    self.assertEqual([1, 1, 1, 1, 1, 1, 1], packs.pack_distribution(7))
    self.assertEqual([1, 1, 1, 1, 1, 1, 1, 1], packs.pack_distribution(8))
    self.assertEqual([1, 1, 1, 1, 1, 1, 1, 1, 1], packs.pack_distribution(9))