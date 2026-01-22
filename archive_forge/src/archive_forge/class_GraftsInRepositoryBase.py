import os
import shutil
import tempfile
from dulwich.tests import TestCase
from ..errors import ObjectFormatException
from ..objects import Tree
from ..repo import MemoryRepo, Repo, parse_graftpoints, serialize_graftpoints
class GraftsInRepositoryBase:

    def tearDown(self):
        super().tearDown()

    def get_repo_with_grafts(self, grafts):
        r = self._repo
        r._add_graftpoints(grafts)
        return r

    def test_no_grafts(self):
        r = self.get_repo_with_grafts({})
        shas = [e.commit.id for e in r.get_walker()]
        self.assertEqual(shas, self._shas[::-1])

    def test_no_parents_graft(self):
        r = self.get_repo_with_grafts({self._repo.head(): []})
        self.assertEqual([e.commit.id for e in r.get_walker()], [r.head()])

    def test_existing_parent_graft(self):
        r = self.get_repo_with_grafts({self._shas[-1]: [self._shas[0]]})
        self.assertEqual([e.commit.id for e in r.get_walker()], [self._shas[-1], self._shas[0]])

    def test_remove_graft(self):
        r = self.get_repo_with_grafts({self._repo.head(): []})
        r._remove_graftpoints([self._repo.head()])
        self.assertEqual([e.commit.id for e in r.get_walker()], self._shas[::-1])

    def test_object_store_fail_invalid_parents(self):
        r = self._repo
        self.assertRaises(ObjectFormatException, r._add_graftpoints, {self._shas[-1]: ['1']})