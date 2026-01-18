from __future__ import annotations
import os
import zipfile
from typing import Union
from twisted.python.filepath import _coerceToFilesystemEncoding
from twisted.python.zippath import ZipArchive, ZipPath
from twisted.test.test_paths import AbstractFilePathTests
def test_zipPathReprParentDirSegment(self) -> None:
    """
        The repr of a ZipPath with C{".."} in the internal part of its path
        includes the C{".."} rather than applying the usual parent directory
        meaning.
        """
    child = self.path.child('foo').child('..').child('bar')
    pathRepr = 'ZipPath(%r)' % (self.nativecmn + '.zip' + os.sep.join(['', 'foo', '..', 'bar']))
    self.assertEqual(repr(child), pathRepr)