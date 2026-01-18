import os
import platform
import shutil
import tempfile
import unittest
from gzip import GzipFile
from pathlib import Path
import pytest
from monty.shutil import (
def test_recursive_copy_and_compress(self):
    copy_r(os.path.join(test_dir, 'cpr_src'), os.path.join(test_dir, 'cpr_dst'))
    assert os.path.exists(os.path.join(test_dir, 'cpr_dst', 'test'))
    assert os.path.exists(os.path.join(test_dir, 'cpr_dst', 'sub', 'testr'))
    compress_dir(os.path.join(test_dir, 'cpr_src'))
    assert os.path.exists(os.path.join(test_dir, 'cpr_src', 'test.gz'))
    assert os.path.exists(os.path.join(test_dir, 'cpr_src', 'sub', 'testr.gz'))
    decompress_dir(os.path.join(test_dir, 'cpr_src'))
    assert os.path.exists(os.path.join(test_dir, 'cpr_src', 'test'))
    assert os.path.exists(os.path.join(test_dir, 'cpr_src', 'sub', 'testr'))
    with open(os.path.join(test_dir, 'cpr_src', 'test')) as f:
        txt = f.read()
        assert txt == 'what'