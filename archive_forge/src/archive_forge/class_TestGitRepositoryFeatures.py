import os
import dulwich
from dulwich.repo import Repo as GitRepo
from ... import config, errors, revision
from ...repository import InterRepository, Repository
from .. import dir, repository, tests
from ..mapping import default_mapping
from ..object_store import BazaarObjectStore
from ..push import MissingObjectsIterator
class TestGitRepositoryFeatures(tests.TestCaseInTempDir):
    """Feature tests for GitRepository."""

    def _do_commit(self):
        builder = tests.GitBranchBuilder()
        builder.set_file(b'a', b'text for a\n', False)
        commit_handle = builder.commit(b'Joe Foo <joe@foo.com>', b'message')
        mapping = builder.finish()
        return mapping[commit_handle]

    def test_open_existing(self):
        GitRepo.init(self.test_dir)
        repo = Repository.open('.')
        self.assertIsInstance(repo, repository.GitRepository)

    def test_has_git_repo(self):
        GitRepo.init(self.test_dir)
        repo = Repository.open('.')
        self.assertIsInstance(repo._git, dulwich.repo.BaseRepo)

    def test_has_revision(self):
        GitRepo.init(self.test_dir)
        commit_id = self._do_commit()
        repo = Repository.open('.')
        self.assertFalse(repo.has_revision(b'foobar'))
        revid = default_mapping.revision_id_foreign_to_bzr(commit_id)
        self.assertTrue(repo.has_revision(revid))

    def test_has_revisions(self):
        GitRepo.init(self.test_dir)
        commit_id = self._do_commit()
        repo = Repository.open('.')
        self.assertEqual(set(), repo.has_revisions([b'foobar']))
        revid = default_mapping.revision_id_foreign_to_bzr(commit_id)
        self.assertEqual({revid}, repo.has_revisions([b'foobar', revid]))

    def test_get_revision(self):
        GitRepo.init(self.test_dir)
        commit_id = self._do_commit()
        revid = default_mapping.revision_id_foreign_to_bzr(commit_id)
        repo = Repository.open('.')
        rev = repo.get_revision(revid)
        self.assertIsInstance(rev, revision.Revision)

    def test_get_revision_unknown(self):
        GitRepo.init(self.test_dir)
        repo = Repository.open('.')
        self.assertRaises(errors.NoSuchRevision, repo.get_revision, b'bla')

    def simple_commit(self):
        GitRepo.init(self.test_dir)
        builder = tests.GitBranchBuilder()
        builder.set_file(b'data', b'text\n', False)
        builder.set_file(b'executable', b'content', True)
        builder.set_symlink(b'link', b'broken')
        builder.set_file(b'subdir/subfile', b'subdir text\n', False)
        commit_handle = builder.commit(b'Joe Foo <joe@foo.com>', b'message', timestamp=1205433193)
        mapping = builder.finish()
        return mapping[commit_handle]

    def test_pack(self):
        commit_id = self.simple_commit()
        repo = Repository.open('.')
        repo.pack()

    def test_unlock_closes(self):
        commit_id = self.simple_commit()
        repo = Repository.open('.')
        repo.pack()
        with repo.lock_read():
            repo.all_revision_ids()
            self.assertTrue(len(repo._git.object_store._pack_cache) > 0)
        self.assertEqual(len(repo._git.object_store._pack_cache), 0)

    def test_revision_tree(self):
        commit_id = self.simple_commit()
        revid = default_mapping.revision_id_foreign_to_bzr(commit_id)
        repo = Repository.open('.')
        tree = repo.revision_tree(revid)
        self.assertEqual(tree.get_revision_id(), revid)
        self.assertEqual(b'text\n', tree.get_file_text('data'))