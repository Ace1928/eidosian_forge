from dulwich.tests import TestCase
from ..objects import Blob
from ..objectspec import (
from ..repo import MemoryRepo
from .utils import build_commit_graph
class ParseObjectTests(TestCase):
    """Test parse_object."""

    def test_nonexistent(self):
        r = MemoryRepo()
        self.assertRaises(KeyError, parse_object, r, 'thisdoesnotexist')

    def test_blob_by_sha(self):
        r = MemoryRepo()
        b = Blob.from_string(b'Blah')
        r.object_store.add_object(b)
        self.assertEqual(b, parse_object(r, b.id))