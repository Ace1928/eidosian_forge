import struct
import tarfile
from io import BytesIO
from unittest import skipUnless
from dulwich.tests import TestCase
from ..archive import tar_stream
from ..object_store import MemoryObjectStore
from ..objects import Blob, Tree
from .utils import build_commit_graph
def test_prefix(self):
    stream = self._get_example_tar_stream(mtime=0, prefix=b'blah')
    tf = tarfile.TarFile(fileobj=stream)
    self.addCleanup(tf.close)
    self.assertEqual(['blah/somename'], tf.getnames())