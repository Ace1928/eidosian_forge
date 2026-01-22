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
class TestRepositoryConverter(TestCaseWithTransport):

    def test_convert_empty(self):
        source_format = TestRepositoryFormat1()
        target_format = TestRepositoryFormat2()
        repository.format_registry.register(source_format)
        self.addCleanup(repository.format_registry.remove, source_format)
        repository.format_registry.register(target_format)
        self.addCleanup(repository.format_registry.remove, target_format)
        t = self.get_transport()
        t.mkdir('repository')
        repo_dir = bzrdir.BzrDirMetaFormat1().initialize('repository')
        repo = TestRepositoryFormat1().initialize(repo_dir)
        converter = repository.CopyConverter(target_format)
        with breezy.ui.ui_factory.nested_progress_bar() as pb:
            converter.convert(repo, pb)
        repo = repo_dir.open_repository()
        self.assertTrue(isinstance(target_format, repo._format.__class__))