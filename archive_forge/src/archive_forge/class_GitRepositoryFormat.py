import os
import dulwich
from dulwich.repo import Repo as GitRepo
from ... import config, errors, revision
from ...repository import InterRepository, Repository
from .. import dir, repository, tests
from ..mapping import default_mapping
from ..object_store import BazaarObjectStore
from ..push import MissingObjectsIterator
class GitRepositoryFormat(tests.TestCase):

    def setUp(self):
        super().setUp()
        self.format = repository.GitRepositoryFormat()

    def test_get_format_description(self):
        self.assertEqual('Git Repository', self.format.get_format_description())