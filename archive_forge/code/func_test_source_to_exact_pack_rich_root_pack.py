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
def test_source_to_exact_pack_rich_root_pack(self):
    source = self.make_repository('source', format='rich-root-pack')
    target = self.make_repository('target', format='rich-root-pack')
    stream_source = source._get_source(target._format)
    self.assertIsInstance(stream_source, knitpack_repo.KnitPackStreamSource)