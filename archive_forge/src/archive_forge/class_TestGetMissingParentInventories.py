import sys
from breezy import branch, controldir, errors, memorytree, tests
from breezy.bzr import branch as bzrbranch
from breezy.bzr import remote, versionedfile
from breezy.bzr.tests.per_repository_vf import (
from breezy.tests.scenarios import load_tests_apply_scenarios
class TestGetMissingParentInventories(TestCaseWithRepository):
    scenarios = all_repository_vf_format_scenarios()

    def test_empty_get_missing_parent_inventories(self):
        """A new write group has no missing parent inventories."""
        repo = self.make_repository('.')
        repo.lock_write()
        repo.start_write_group()
        try:
            self.assertEqual(set(), set(repo.get_missing_parent_inventories()))
        finally:
            repo.commit_write_group()
            repo.unlock()

    def branch_trunk_and_make_tree(self, trunk_repo, relpath):
        tree = self.make_branch_and_memory_tree('branch')
        trunk_repo.lock_read()
        self.addCleanup(trunk_repo.unlock)
        tree.branch.repository.fetch(trunk_repo, revision_id=b'rev-1')
        tree.set_parent_ids([b'rev-1'])
        return tree

    def make_first_commit(self, repo):
        trunk = repo.controldir.create_branch()
        tree = memorytree.MemoryTree.create_on_branch(trunk)
        tree.lock_write()
        tree.add([''], ['directory'], [b'TREE_ROOT'])
        tree.add(['dir'], ['directory'], [b'dir-id'])
        tree.add(['filename'], ['file'], [b'file-id'])
        tree.put_file_bytes_non_atomic('filename', b'content\n')
        tree.commit('Trunk commit', rev_id=b'rev-0')
        tree.commit('Trunk commit', rev_id=b'rev-1')
        tree.unlock()

    def make_new_commit_in_new_repo(self, trunk_repo, parents=None):
        tree = self.branch_trunk_and_make_tree(trunk_repo, 'branch')
        tree.set_parent_ids(parents)
        tree.commit('Branch commit', rev_id=b'rev-2')
        branch_repo = tree.branch.repository
        branch_repo.lock_read()
        self.addCleanup(branch_repo.unlock)
        return branch_repo

    def make_stackable_repo(self, relpath='trunk'):
        if isinstance(self.repository_format, remote.RemoteRepositoryFormat):
            repo = self.make_repository(relpath, format='1.9')
            dir = controldir.ControlDir.open(self.get_url(relpath))
            repo = dir.open_repository()
        else:
            repo = self.make_repository(relpath)
        if not repo._format.supports_external_lookups:
            raise tests.TestNotApplicable('format not stackable')
        repo.controldir._format.set_branch_format(bzrbranch.BzrBranchFormat7())
        return repo

    def reopen_repo_and_resume_write_group(self, repo):
        try:
            resume_tokens = repo.suspend_write_group()
        except errors.UnsuspendableWriteGroup:
            repo.unlock()
            return
        repo.unlock()
        reopened_repo = repo.controldir.open_repository()
        reopened_repo.lock_write()
        self.addCleanup(reopened_repo.unlock)
        reopened_repo.resume_write_group(resume_tokens)
        return reopened_repo

    def test_ghost_revision(self):
        """A parent inventory may be absent if all the needed texts are present.
        i.e., a ghost revision isn't (necessarily) considered to be a missing
        parent inventory.
        """
        trunk_repo = self.make_stackable_repo()
        self.make_first_commit(trunk_repo)
        trunk_repo.lock_read()
        self.addCleanup(trunk_repo.unlock)
        branch_repo = self.make_new_commit_in_new_repo(trunk_repo, parents=[b'rev-1', b'ghost-rev'])
        inv = branch_repo.get_inventory(b'rev-2')
        repo = self.make_stackable_repo('stacked')
        repo.lock_write()
        repo.start_write_group()
        rich_root = branch_repo._format.rich_root_data
        all_texts = [(ie.file_id, ie.revision) for ie in inv.iter_just_entries() if rich_root or inv.id2path(ie.file_id) != '']
        repo.texts.insert_record_stream(branch_repo.texts.get_record_stream(all_texts, 'unordered', False))
        repo.add_inventory(b'rev-2', inv, [b'rev-1', b'ghost-rev'])
        repo.revisions.insert_record_stream(branch_repo.revisions.get_record_stream([(b'rev-2',)], 'unordered', False))
        self.assertEqual(set(), repo.get_missing_parent_inventories())
        reopened_repo = self.reopen_repo_and_resume_write_group(repo)
        self.assertEqual(set(), reopened_repo.get_missing_parent_inventories())
        reopened_repo.abort_write_group()

    def test_get_missing_parent_inventories(self):
        """A stacked repo with a single revision and inventory (no parent
        inventory) in it must have all the texts in its inventory (even if not
        changed w.r.t. to the absent parent), otherwise it will report missing
        texts/parent inventory.

        The core of this test is that a file was changed in rev-1, but in a
        stacked repo that only has rev-2
        """
        trunk_repo = self.make_stackable_repo()
        self.make_first_commit(trunk_repo)
        trunk_repo.lock_read()
        self.addCleanup(trunk_repo.unlock)
        branch_repo = self.make_new_commit_in_new_repo(trunk_repo, parents=[b'rev-1'])
        inv = branch_repo.get_inventory(b'rev-2')
        repo = self.make_stackable_repo('stacked')
        repo.lock_write()
        repo.start_write_group()
        repo.add_inventory(b'rev-2', inv, [b'rev-1'])
        repo.revisions.insert_record_stream(branch_repo.revisions.get_record_stream([(b'rev-2',)], 'unordered', False))
        self.assertEqual(set(), repo.inventories.get_missing_compression_parent_keys())
        self.assertEqual({('inventories', b'rev-1')}, repo.get_missing_parent_inventories())
        reopened_repo = self.reopen_repo_and_resume_write_group(repo)
        self.assertEqual({('inventories', b'rev-1')}, reopened_repo.get_missing_parent_inventories())
        reopened_repo.inventories.insert_record_stream(branch_repo.inventories.get_record_stream([(b'rev-1',)], 'unordered', False))
        self.assertEqual(set(), reopened_repo.get_missing_parent_inventories())
        reopened_repo.abort_write_group()

    def test_get_missing_parent_inventories_check(self):
        builder = self.make_branch_builder('test')
        builder.build_snapshot([b'ghost-parent-id'], [('add', ('', b'root-id', 'directory', None)), ('add', ('file', b'file-id', 'file', b'content\n'))], allow_leftmost_as_ghost=True, revision_id=b'A-id')
        b = builder.get_branch()
        b.lock_read()
        self.addCleanup(b.unlock)
        repo = self.make_repository('test-repo')
        repo.lock_write()
        self.addCleanup(repo.unlock)
        repo.start_write_group()
        self.addCleanup(repo.abort_write_group)
        text_keys = [(b'file-id', b'A-id')]
        if repo.supports_rich_root():
            text_keys.append((b'root-id', b'A-id'))
        repo.texts.insert_record_stream(b.repository.texts.get_record_stream(text_keys, 'unordered', True))
        repo.add_revision(b'A-id', b.repository.get_revision(b'A-id'), b.repository.get_inventory(b'A-id'))
        get_missing = repo.get_missing_parent_inventories
        if repo._format.supports_external_lookups:
            self.assertEqual({('inventories', b'ghost-parent-id')}, get_missing(check_for_missing_texts=False))
            self.assertEqual(set(), get_missing(check_for_missing_texts=True))
            self.assertEqual(set(), get_missing())
        else:
            self.assertEqual(set(), get_missing(check_for_missing_texts=False))
            self.assertEqual(set(), get_missing(check_for_missing_texts=True))
            self.assertEqual(set(), get_missing())

    def test_insert_stream_passes_resume_info(self):
        repo = self.make_repository('test-repo')
        if not repo._format.supports_external_lookups or isinstance(repo, remote.RemoteRepository):
            raise tests.TestNotApplicable('only valid for direct connections to resumable repos')
        call_log = []
        orig = repo.get_missing_parent_inventories

        def get_missing(check_for_missing_texts=True):
            call_log.append(check_for_missing_texts)
            return orig(check_for_missing_texts=check_for_missing_texts)
        repo.get_missing_parent_inventories = get_missing
        repo.lock_write()
        self.addCleanup(repo.unlock)
        sink = repo._get_sink()
        sink.insert_stream((), repo._format, [])
        self.assertEqual([False], call_log)
        del call_log[:]
        repo.start_write_group()
        repo.texts.insert_record_stream([versionedfile.FulltextContentFactory((b'file-id', b'rev-id'), (), None, b'lines\n')])
        tokens = repo.suspend_write_group()
        self.assertNotEqual([], tokens)
        sink.insert_stream((), repo._format, tokens)
        self.assertEqual([True], call_log)

    def test_insert_stream_without_locking_fails_without_lock(self):
        repo = self.make_repository('test-repo')
        sink = repo._get_sink()
        stream = [('texts', [versionedfile.FulltextContentFactory((b'file-id', b'rev-id'), (), None, b'lines\n')])]
        self.assertRaises(errors.ObjectNotLocked, sink.insert_stream_without_locking, stream, repo._format)

    def test_insert_stream_without_locking_fails_without_write_group(self):
        repo = self.make_repository('test-repo')
        self.addCleanup(repo.lock_write().unlock)
        sink = repo._get_sink()
        stream = [('texts', [versionedfile.FulltextContentFactory((b'file-id', b'rev-id'), (), None, b'lines\n')])]
        self.assertRaises(errors.BzrError, sink.insert_stream_without_locking, stream, repo._format)

    def test_insert_stream_without_locking(self):
        repo = self.make_repository('test-repo')
        self.addCleanup(repo.lock_write().unlock)
        repo.start_write_group()
        sink = repo._get_sink()
        stream = [('texts', [versionedfile.FulltextContentFactory((b'file-id', b'rev-id'), (), None, b'lines\n')])]
        missing_keys = sink.insert_stream_without_locking(stream, repo._format)
        repo.commit_write_group()
        self.assertEqual(set(), missing_keys)