from __future__ import annotations
import os
import zipfile
from typing import Union
from twisted.python.filepath import _coerceToFilesystemEncoding
from twisted.python.zippath import ZipArchive, ZipPath
from twisted.test.test_paths import AbstractFilePathTests

        Make sure that invoking ZipArchive's repr prints the correct class
        name and an absolute path to the zip file.
        