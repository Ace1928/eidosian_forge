from dulwich.tests import TestCase
from ..object_store import MemoryObjectStore, MissingObjectFinder
from ..objects import Blob
from .utils import build_commit_graph, make_object, make_tag
class MissingObjectFinderTest(TestCase):

    def setUp(self):
        super().setUp()
        self.store = MemoryObjectStore()
        self.commits = []

    def cmt(self, n):
        return self.commits[n - 1]

    def assertMissingMatch(self, haves, wants, expected):
        for sha, path in MissingObjectFinder(self.store, haves, wants, shallow=set()):
            self.assertIn(sha, expected, f'({sha},{path}) erroneously reported as missing')
            expected.remove(sha)
        self.assertEqual(len(expected), 0, f'some objects are not reported as missing: {expected}')