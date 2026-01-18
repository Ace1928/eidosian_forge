import os
import dulwich
from dulwich.repo import Repo as GitRepo
from ... import config, errors, revision
from ...repository import InterRepository, Repository
from .. import dir, repository, tests
from ..mapping import default_mapping
from ..object_store import BazaarObjectStore
from ..push import MissingObjectsIterator
def test_revision_tree(self):
    commit_id = self.simple_commit()
    revid = default_mapping.revision_id_foreign_to_bzr(commit_id)
    repo = Repository.open('.')
    tree = repo.revision_tree(revid)
    self.assertEqual(tree.get_revision_id(), revid)
    self.assertEqual(b'text\n', tree.get_file_text('data'))