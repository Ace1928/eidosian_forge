import shutil
import tempfile
from ..lfs import LFSStore
from . import TestCase
class LFSTests(TestCase):

    def setUp(self):
        super().setUp()
        self.test_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.test_dir)
        self.lfs = LFSStore.create(self.test_dir)

    def test_create(self):
        sha = self.lfs.write_object([b'a', b'b'])
        with self.lfs.open_object(sha) as f:
            self.assertEqual(b'ab', f.read())

    def test_missing(self):
        self.assertRaises(KeyError, self.lfs.open_object, 'abcdeabcdeabcdeabcde')