from dulwich.tests import TestCase
from ..object_store import MemoryObjectStore, MissingObjectFinder
from ..objects import Blob
from .utils import build_commit_graph, make_object, make_tag
def test_tagged_tagged_blob(self):
    self.assertMissingMatch([], [self._tag_of_tag_of_blob.id], [self._tag_of_tag_of_blob.id, self._tag_of_blob.id, self.f1_1_id])