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
def test_pack_distribution_stable_at_boundaries(self):
    """When there are multi-rev packs the counts are stable."""
    packs = self.get_packs()
    self.assertEqual([10], packs.pack_distribution(10))
    self.assertEqual([10, 1], packs.pack_distribution(11))
    self.assertEqual([10, 10], packs.pack_distribution(20))
    self.assertEqual([10, 10, 1], packs.pack_distribution(21))
    self.assertEqual([100], packs.pack_distribution(100))
    self.assertEqual([100, 1], packs.pack_distribution(101))
    self.assertEqual([100, 10, 1], packs.pack_distribution(111))
    self.assertEqual([100, 100], packs.pack_distribution(200))
    self.assertEqual([100, 100, 1], packs.pack_distribution(201))
    self.assertEqual([100, 100, 10, 1], packs.pack_distribution(211))