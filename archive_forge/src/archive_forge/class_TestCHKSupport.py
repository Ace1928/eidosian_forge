from breezy import errors, osutils, repository
from breezy.bzr import btree_index
from breezy.bzr.remote import RemoteRepository
from breezy.bzr.tests.per_repository_chk import TestCaseWithRepositoryCHK
from breezy.bzr.versionedfile import VersionedFiles
from breezy.tests import TestNotApplicable
class TestCHKSupport(TestCaseWithRepositoryCHK):

    def test_chk_bytes_attribute_is_VersionedFiles(self):
        repo = self.make_repository('.')
        self.assertIsInstance(repo.chk_bytes, VersionedFiles)

    def test_add_bytes_to_chk_bytes_store(self):
        repo = self.make_repository('.')
        with repo.lock_write(), repository.WriteGroup(repo):
            sha1, len, _ = repo.chk_bytes.add_lines((None,), None, [b'foo\n', b'bar\n'], random_id=True)
            self.assertEqual(b'4e48e2c9a3d2ca8a708cb0cc545700544efb5021', sha1)
            self.assertEqual({(b'sha1:4e48e2c9a3d2ca8a708cb0cc545700544efb5021',)}, repo.chk_bytes.keys())
        with repo.lock_read():
            self.assertEqual({(b'sha1:4e48e2c9a3d2ca8a708cb0cc545700544efb5021',)}, repo.chk_bytes.keys())
        repo = repo.controldir.open_repository()
        with repo.lock_read():
            self.assertEqual({(b'sha1:4e48e2c9a3d2ca8a708cb0cc545700544efb5021',)}, repo.chk_bytes.keys())

    def test_pack_preserves_chk_bytes_store(self):
        leaf_lines = [b'chkleaf:\n', b'0\n', b'1\n', b'0\n', b'\n']
        leaf_sha1 = osutils.sha_strings(leaf_lines)
        node_lines = [b'chknode:\n', b'0\n', b'1\n', b'1\n', b'foo\n', b'\x00sha1:%s\n' % (leaf_sha1,)]
        node_sha1 = osutils.sha_strings(node_lines)
        expected_set = {(b'sha1:' + leaf_sha1,), (b'sha1:' + node_sha1,)}
        repo = self.make_repository('.')
        with repo.lock_write():
            with repository.WriteGroup(repo):
                repo.chk_bytes.add_lines((None,), None, node_lines, random_id=True)
            with repository.WriteGroup(repo):
                repo.chk_bytes.add_lines((None,), None, leaf_lines, random_id=True)
            repo.pack()
            self.assertEqual(expected_set, repo.chk_bytes.keys())
        repo = repo.controldir.open_repository()
        with repo.lock_read():
            self.assertEqual(expected_set, repo.chk_bytes.keys())

    def test_chk_bytes_are_fully_buffered(self):
        repo = self.make_repository('.')
        repo.lock_write()
        self.addCleanup(repo.unlock)
        with repository.WriteGroup(repo):
            sha1, len, _ = repo.chk_bytes.add_lines((None,), None, [b'foo\n', b'bar\n'], random_id=True)
            self.assertEqual(b'4e48e2c9a3d2ca8a708cb0cc545700544efb5021', sha1)
            self.assertEqual({(b'sha1:4e48e2c9a3d2ca8a708cb0cc545700544efb5021',)}, repo.chk_bytes.keys())
        index = repo.chk_bytes._index._graph_index._indices[0]
        self.assertIsInstance(index, btree_index.BTreeGraphIndex)
        self.assertIs(type(index._leaf_node_cache), dict)
        repo2 = repository.Repository.open(self.get_url())
        repo2.lock_read()
        self.addCleanup(repo2.unlock)
        index = repo2.chk_bytes._index._graph_index._indices[0]
        self.assertIsInstance(index, btree_index.BTreeGraphIndex)
        self.assertIs(type(index._leaf_node_cache), dict)