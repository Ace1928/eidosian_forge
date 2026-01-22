import random
import struct
import zipfile
from hashlib import md5
from twisted.python import filepath, zipstream
from twisted.trial import unittest
class ZipFileEntryTests(FileEntryMixin, unittest.TestCase):
    """
    ZipFileEntry should be file-like
    """
    compression = zipfile.ZIP_STORED