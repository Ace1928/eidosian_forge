from breezy import errors, gpg
from breezy import repository as _mod_repository
from breezy import revision as _mod_revision
from breezy import tests
from breezy.bzr import inventory, versionedfile, vf_repository
from breezy.bzr.tests.per_repository_vf import (
from breezy.tests.matchers import MatchesAncestry
from breezy.tests.scenarios import load_tests_apply_scenarios
class TestRepository(TestCaseWithRepository):
    scenarios = all_repository_vf_format_scenarios()

    def assertFormatAttribute(self, attribute, allowed_values):
        """Assert that the format has an attribute 'attribute'."""
        repo = self.make_repository('repo')
        self.assertSubset([getattr(repo._format, attribute)], allowed_values)

    def test_attribute__fetch_order(self):
        """Test the _fetch_order attribute."""
        self.assertFormatAttribute('_fetch_order', ('topological', 'unordered'))

    def test_attribute__fetch_uses_deltas(self):
        """Test the _fetch_uses_deltas attribute."""
        self.assertFormatAttribute('_fetch_uses_deltas', (True, False))

    def test_attribute_inventories_store(self):
        """Test the existence of the inventories attribute."""
        tree = self.make_branch_and_tree('tree')
        repo = tree.branch.repository
        self.assertIsInstance(repo.inventories, versionedfile.VersionedFiles)

    def test_attribute_inventories_basics(self):
        """Test basic aspects of the inventories attribute."""
        tree = self.make_branch_and_tree('tree')
        repo = tree.branch.repository
        rev_id = (tree.commit('a'),)
        tree.lock_read()
        self.addCleanup(tree.unlock)
        self.assertEqual({rev_id}, set(repo.inventories.keys()))

    def test_attribute_revision_store(self):
        """Test the existence of the revisions attribute."""
        tree = self.make_branch_and_tree('tree')
        repo = tree.branch.repository
        self.assertIsInstance(repo.revisions, versionedfile.VersionedFiles)

    def test_attribute_revision_store_basics(self):
        """Test the basic behaviour of the revisions attribute."""
        tree = self.make_branch_and_tree('tree')
        repo = tree.branch.repository
        repo.lock_write()
        try:
            self.assertEqual(set(), set(repo.revisions.keys()))
            revid = (tree.commit('foo'),)
            self.assertEqual({revid}, set(repo.revisions.keys()))
            self.assertEqual({revid: ()}, repo.revisions.get_parent_map([revid]))
        finally:
            repo.unlock()
        tree2 = self.make_branch_and_tree('tree2')
        tree2.pull(tree.branch)
        left_id = (tree2.commit('left'),)
        right_id = (tree.commit('right'),)
        tree.merge_from_branch(tree2.branch)
        merge_id = (tree.commit('merged'),)
        repo.lock_read()
        self.addCleanup(repo.unlock)
        self.assertEqual({revid, left_id, right_id, merge_id}, set(repo.revisions.keys()))
        self.assertEqual({revid: (), left_id: (revid,), right_id: (revid,), merge_id: (right_id, left_id)}, repo.revisions.get_parent_map(repo.revisions.keys()))

    def test_attribute_signature_store(self):
        """Test the existence of the signatures attribute."""
        tree = self.make_branch_and_tree('tree')
        repo = tree.branch.repository
        self.assertIsInstance(repo.signatures, versionedfile.VersionedFiles)

    def test_exposed_versioned_files_are_marked_dirty(self):
        repo = self.make_repository('.')
        repo.lock_write()
        signatures = repo.signatures
        revisions = repo.revisions
        inventories = repo.inventories
        repo.unlock()
        self.assertRaises(errors.ObjectNotLocked, signatures.keys)
        self.assertRaises(errors.ObjectNotLocked, revisions.keys)
        self.assertRaises(errors.ObjectNotLocked, inventories.keys)
        self.assertRaises(errors.ObjectNotLocked, signatures.add_lines, ('foo',), [], [])
        self.assertRaises(errors.ObjectNotLocked, revisions.add_lines, ('foo',), [], [])
        self.assertRaises(errors.ObjectNotLocked, inventories.add_lines, ('foo',), [], [])

    def test__get_sink(self):
        repo = self.make_repository('repo')
        sink = repo._get_sink()
        self.assertIsInstance(sink, vf_repository.StreamSink)

    def test_get_serializer_format(self):
        repo = self.make_repository('.')
        format = repo.get_serializer_format()
        self.assertEqual(repo._serializer.format_num, format)

    def test_add_revision_inventory_sha1(self):
        inv = inventory.Inventory(revision_id=b'A')
        inv.root.revision = b'A'
        inv.root.file_id = b'fixed-root'
        reference_repo = self.make_repository('reference_repo')
        reference_repo.lock_write()
        reference_repo.start_write_group()
        inv_sha1 = reference_repo.add_inventory(b'A', inv, [])
        reference_repo.abort_write_group()
        reference_repo.unlock()
        repo = self.make_repository('repo')
        repo.lock_write()
        repo.start_write_group()
        root_id = inv.root.file_id
        repo.texts.add_lines((b'fixed-root', b'A'), [], [])
        repo.add_revision(b'A', _mod_revision.Revision(b'A', committer='B', timestamp=0, timezone=0, message='C'), inv=inv)
        repo.commit_write_group()
        repo.unlock()
        repo.lock_read()
        self.assertEqual(inv_sha1, repo.get_revision(b'A').inventory_sha1)
        repo.unlock()

    def test_install_revisions(self):
        wt = self.make_branch_and_tree('source')
        wt.commit('A', allow_pointless=True, rev_id=b'A')
        repo = wt.branch.repository
        repo.lock_write()
        repo.start_write_group()
        repo.sign_revision(b'A', gpg.LoopbackGPGStrategy(None))
        repo.commit_write_group()
        repo.unlock()
        repo.lock_read()
        self.addCleanup(repo.unlock)
        repo2 = self.make_repository('repo2')
        revision = repo.get_revision(b'A')
        tree = repo.revision_tree(b'A')
        signature = repo.get_signature_text(b'A')
        repo2.lock_write()
        self.addCleanup(repo2.unlock)
        vf_repository.install_revisions(repo2, [(revision, tree, signature)])
        self.assertEqual(revision, repo2.get_revision(b'A'))
        self.assertEqual(signature, repo2.get_signature_text(b'A'))

    def test_attribute_text_store(self):
        """Test the existence of the texts attribute."""
        tree = self.make_branch_and_tree('tree')
        repo = tree.branch.repository
        self.assertIsInstance(repo.texts, versionedfile.VersionedFiles)

    def test_iter_inventories_is_ordered(self):
        tree = self.make_branch_and_tree('a')
        first_revision = tree.commit('')
        second_revision = tree.commit('')
        tree.lock_read()
        self.addCleanup(tree.unlock)
        revs = (first_revision, second_revision)
        invs = tree.branch.repository.iter_inventories(revs)
        for rev_id, inv in zip(revs, invs):
            self.assertEqual(rev_id, inv.revision_id)
            self.assertIsInstance(inv, inventory.CommonInventory)

    def test_item_keys_introduced_by(self):
        tree = self.make_branch_and_tree('t')
        self.build_tree(['t/foo'])
        tree.add('foo', ids=b'file1')
        tree.commit('message', rev_id=b'rev_id')
        repo = tree.branch.repository
        repo.lock_write()
        repo.start_write_group()
        try:
            repo.sign_revision(b'rev_id', gpg.LoopbackGPGStrategy(None))
        except errors.UnsupportedOperation:
            signature_texts = []
        else:
            signature_texts = [b'rev_id']
        repo.commit_write_group()
        repo.unlock()
        repo.lock_read()
        self.addCleanup(repo.unlock)
        expected_item_keys = [('file', b'file1', [b'rev_id']), ('inventory', None, [b'rev_id']), ('signatures', None, signature_texts), ('revisions', None, [b'rev_id'])]
        item_keys = list(repo.item_keys_introduced_by([b'rev_id']))
        item_keys = [(kind, file_id, list(versions)) for kind, file_id, versions in item_keys]
        if repo.supports_rich_root():
            inv = repo.get_inventory(b'rev_id')
            root_item_key = ('file', inv.root.file_id, [b'rev_id'])
            self.assertIn(root_item_key, item_keys)
            item_keys.remove(root_item_key)
        self.assertEqual(expected_item_keys, item_keys)

    def test_attribute_text_store_basics(self):
        """Test the basic behaviour of the text store."""
        tree = self.make_branch_and_tree('tree')
        repo = tree.branch.repository
        file_id = b'Foo:Bar'
        file_key = (file_id,)
        with tree.lock_write():
            self.assertEqual(set(), set(repo.texts.keys()))
            tree.add(['foo'], ['file'], [file_id])
            tree.put_file_bytes_non_atomic('foo', b'content\n')
            try:
                rev_key = (tree.commit('foo'),)
            except errors.IllegalPath:
                raise tests.TestNotApplicable('file_id %r cannot be stored on this platform for this repo format' % (file_id,))
            if repo._format.rich_root_data:
                root_commit = (tree.path2id(''),) + rev_key
                keys = {root_commit}
                parents = {root_commit: ()}
            else:
                keys = set()
                parents = {}
            keys.add(file_key + rev_key)
            parents[file_key + rev_key] = ()
            self.assertEqual(keys, set(repo.texts.keys()))
            self.assertEqual(parents, repo.texts.get_parent_map(repo.texts.keys()))
        tree2 = self.make_branch_and_tree('tree2')
        tree2.pull(tree.branch)
        tree2.put_file_bytes_non_atomic('foo', b'right\n')
        right_key = (tree2.commit('right'),)
        keys.add(file_key + right_key)
        parents[file_key + right_key] = (file_key + rev_key,)
        tree.put_file_bytes_non_atomic('foo', b'left\n')
        left_key = (tree.commit('left'),)
        keys.add(file_key + left_key)
        parents[file_key + left_key] = (file_key + rev_key,)
        tree.merge_from_branch(tree2.branch)
        tree.put_file_bytes_non_atomic('foo', b'merged\n')
        try:
            tree.auto_resolve()
        except errors.UnsupportedOperation:
            pass
        merge_key = (tree.commit('merged'),)
        keys.add(file_key + merge_key)
        parents[file_key + merge_key] = (file_key + left_key, file_key + right_key)
        repo.lock_read()
        self.addCleanup(repo.unlock)
        self.assertEqual(keys, set(repo.texts.keys()))
        self.assertEqual(parents, repo.texts.get_parent_map(repo.texts.keys()))