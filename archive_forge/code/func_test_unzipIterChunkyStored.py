import random
import struct
import zipfile
from hashlib import md5
from twisted.python import filepath, zipstream
from twisted.trial import unittest
def test_unzipIterChunkyStored(self):
    """
        unzipIterChunky should unzip the given number of bytes per iteration on
        a stored archive.
        """
    self._unzipIterChunkyTest(zipfile.ZIP_STORED, 500, 35, 45)