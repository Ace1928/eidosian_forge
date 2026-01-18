import struct
import tarfile
from io import BytesIO
from unittest import skipUnless
from dulwich.tests import TestCase
from ..archive import tar_stream
from ..object_store import MemoryObjectStore
from ..objects import Blob, Tree
from .utils import build_commit_graph
@skipUnless(patch, 'Required mock.patch')
def test_same_file(self):
    contents = [None, None]
    for format in ['', 'gz', 'bz2']:
        for i in [0, 1]:
            with patch('time.time', return_value=i):
                stream = self._get_example_tar_stream(mtime=0, format=format)
                contents[i] = stream.getvalue()
        self.assertEqual(contents[0], contents[1], 'Different file contents for format %r' % format)