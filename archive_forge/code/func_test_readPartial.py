import random
import struct
import zipfile
from hashlib import md5
from twisted.python import filepath, zipstream
from twisted.trial import unittest
def test_readPartial(self):
    """
        C{.read(num)} should read num bytes from the file.
        """
    contents = '0123456789'
    with self.getFileEntry(contents) as entry:
        one = entry.read(4)
        two = entry.read(200)
    self.assertEqual(one, b'0123')
    self.assertEqual(two, b'456789')