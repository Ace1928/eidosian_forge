import os
import dulwich
from dulwich.repo import Repo as GitRepo
from ... import config, errors, revision
from ...repository import InterRepository, Repository
from .. import dir, repository, tests
from ..mapping import default_mapping
from ..object_store import BazaarObjectStore
from ..push import MissingObjectsIterator
def test_unlock_closes(self):
    commit_id = self.simple_commit()
    repo = Repository.open('.')
    repo.pack()
    with repo.lock_read():
        repo.all_revision_ids()
        self.assertTrue(len(repo._git.object_store._pack_cache) > 0)
    self.assertEqual(len(repo._git.object_store._pack_cache), 0)