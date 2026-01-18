import shutil
from base64 import b64decode
from twisted.persisted import dirdbm
from twisted.python import rebuild
from twisted.python.filepath import FilePath
from twisted.trial import unittest
def test_rebuildInteraction(self) -> None:
    s = dirdbm.Shelf('dirdbm.rebuild.test')
    s[b'key'] = b'value'
    rebuild.rebuild(dirdbm)