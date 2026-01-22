import random
import struct
import zipfile
from hashlib import md5
from twisted.python import filepath, zipstream
from twisted.trial import unittest
class DeflatedZipFileEntryTests(FileEntryMixin, unittest.TestCase):
    """
    DeflatedZipFileEntry should be file-like
    """
    compression = zipfile.ZIP_DEFLATED