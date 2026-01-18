import random
import struct
import zipfile
from hashlib import md5
from twisted.python import filepath, zipstream
from twisted.trial import unittest
def test_readWhole(self):
    """
        C{.read()} should read the entire file.
        """
    contents = b'Hello, world!'
    with self.getFileEntry(contents) as entry:
        self.assertEqual(entry.read(), contents)