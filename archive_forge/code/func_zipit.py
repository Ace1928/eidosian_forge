from __future__ import annotations
import os
import zipfile
from typing import Union
from twisted.python.filepath import _coerceToFilesystemEncoding
from twisted.python.zippath import ZipArchive, ZipPath
from twisted.test.test_paths import AbstractFilePathTests
def zipit(dirname: str | bytes, zfname: str | bytes) -> None:
    """
    Create a zipfile on zfname, containing the contents of dirname'
    """
    coercedDirname = _coerceToFilesystemEncoding('', dirname)
    coercedZfname = _coerceToFilesystemEncoding('', zfname)
    with zipfile.ZipFile(coercedZfname, 'w') as zf:
        for root, ignored, files in os.walk(coercedDirname):
            for fname in files:
                fspath = os.path.join(root, fname)
                arcpath = os.path.join(root, fname)[len(dirname) + 1:]
                zf.write(fspath, arcpath)