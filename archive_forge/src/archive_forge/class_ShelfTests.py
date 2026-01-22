import shutil
from base64 import b64decode
from twisted.persisted import dirdbm
from twisted.python import rebuild
from twisted.python.filepath import FilePath
from twisted.trial import unittest
class ShelfTests(DirDbmTests):

    def setUp(self) -> None:
        self.path = FilePath(self.mktemp())
        self.dbm = dirdbm.Shelf(self.path.path)
        self.items = ((b'abc', b'foo'), (b'/lalal', b'\x00\x01'), (b'\x00\n', b'baz'), (b'int', 12), (b'float', 12.0), (b'tuple', (None, 12)))