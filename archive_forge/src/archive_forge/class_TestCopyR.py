import os
import platform
import shutil
import tempfile
import unittest
from gzip import GzipFile
from pathlib import Path
import pytest
from monty.shutil import (
class TestCopyR:

    def setup_method(self):
        os.mkdir(os.path.join(test_dir, 'cpr_src'))
        with open(os.path.join(test_dir, 'cpr_src', 'test'), 'w') as f:
            f.write('what')
        os.mkdir(os.path.join(test_dir, 'cpr_src', 'sub'))
        with open(os.path.join(test_dir, 'cpr_src', 'sub', 'testr'), 'w') as f:
            f.write('what2')

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

    def test_pathlib(self):
        test_path = Path(test_dir)
        copy_r(test_path / 'cpr_src', test_path / 'cpr_dst')
        assert (test_path / 'cpr_dst' / 'test').exists()
        assert (test_path / 'cpr_dst' / 'sub' / 'testr').exists()

    def teardown_method(self):
        shutil.rmtree(os.path.join(test_dir, 'cpr_src'))
        shutil.rmtree(os.path.join(test_dir, 'cpr_dst'))