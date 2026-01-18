from __future__ import annotations
import os
import zipfile
from typing import Union
from twisted.python.filepath import _coerceToFilesystemEncoding
from twisted.python.zippath import ZipArchive, ZipPath
from twisted.test.test_paths import AbstractFilePathTests
def test_zipArchiveRepr(self) -> None:
    """
        Make sure that invoking ZipArchive's repr prints the correct class
        name and an absolute path to the zip file.
        """
    path = ZipArchive(self.nativecmn + '.zip')
    pathRepr = 'ZipArchive({!r})'.format(os.path.abspath(self.nativecmn + '.zip'))
    self.assertEqual(repr(path), pathRepr)
    relativeCommon = self.nativecmn.replace(os.getcwd() + os.sep, '', 1) + '.zip'
    relpath = ZipArchive(relativeCommon)
    self.assertEqual(repr(relpath), pathRepr)