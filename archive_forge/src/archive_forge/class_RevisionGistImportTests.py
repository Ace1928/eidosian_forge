import os
import dulwich
from dulwich.repo import Repo as GitRepo
from ... import config, errors, revision
from ...repository import InterRepository, Repository
from .. import dir, repository, tests
from ..mapping import default_mapping
from ..object_store import BazaarObjectStore
from ..push import MissingObjectsIterator
class RevisionGistImportTests(tests.TestCaseWithTransport):

    def setUp(self):
        tests.TestCaseWithTransport.setUp(self)
        self.git_path = os.path.join(self.test_dir, 'git')
        os.mkdir(self.git_path)
        dulwich.repo.Repo.create(self.git_path)
        self.git_repo = Repository.open(self.git_path)
        self.bzr_tree = self.make_branch_and_tree('bzr')

    def get_inter(self):
        return InterRepository.get(self.bzr_tree.branch.repository, self.git_repo)

    def object_iter(self):
        store = BazaarObjectStore(self.bzr_tree.branch.repository, default_mapping)
        store_iterator = MissingObjectsIterator(store, self.bzr_tree.branch.repository)
        return (store, store_iterator)

    def import_rev(self, revid, parent_lookup=None):
        store, store_iter = self.object_iter()
        store._cache.idmap.start_write_group()
        try:
            return store_iter.import_revision(revid, lossy=True)
        except:
            store._cache.idmap.abort_write_group()
            raise
        else:
            store._cache.idmap.commit_write_group()

    def test_pointless(self):
        revid = self.bzr_tree.commit('pointless', timestamp=1205433193, timezone=0, committer='Jelmer Vernooij <jelmer@samba.org>')
        self.assertEqual(b'2caa8094a5b794961cd9bf582e3e2bb090db0b14', self.import_rev(revid))
        self.assertEqual(b'2caa8094a5b794961cd9bf582e3e2bb090db0b14', self.import_rev(revid))