import os
import platform
import shutil
import tempfile
import unittest
from gzip import GzipFile
from pathlib import Path
import pytest
from monty.shutil import (
@unittest.skipIf(platform.system() == 'Windows', 'Skip on windows')
def test_remove_symlink(self):
    tempdir = tempfile.mkdtemp(dir=test_dir)
    tempf = tempfile.mkstemp(dir=tempdir)[1]
    os.symlink(tempdir, os.path.join(test_dir, 'temp_link'))
    templink = os.path.join(test_dir, 'temp_link')
    remove(templink)
    assert os.path.isfile(tempf)
    assert os.path.isdir(tempdir)
    assert not os.path.islink(templink)
    remove(tempdir)