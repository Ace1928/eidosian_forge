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
class DummyRepository:
    """A dummy repository for testing."""
    _format = None
    _serializer = None

    def supports_rich_root(self):
        if self._format is not None:
            return self._format.rich_root_data
        return False

    def get_graph(self):
        raise NotImplementedError

    def get_parent_map(self, revision_ids):
        raise NotImplementedError