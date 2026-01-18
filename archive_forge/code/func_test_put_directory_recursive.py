from hashlib import md5
from itertools import product
import pytest
from fsspec.tests.abstract.common import GLOB_EDGE_CASES_TESTS
def test_put_directory_recursive(self, fs, fs_join, fs_target, local_fs, local_join, local_path):
    src = local_join(local_path, 'src')
    src_file = local_join(src, 'file')
    local_fs.mkdir(src)
    local_fs.touch(src_file)
    target = fs_target
    assert not fs.exists(target)
    for loop in range(2):
        fs.put(src, target, recursive=True)
        assert fs.isdir(target)
        if loop == 0:
            assert fs.isfile(fs_join(target, 'file'))
            assert not fs.exists(fs_join(target, 'src'))
        else:
            assert fs.isfile(fs_join(target, 'file'))
            assert fs.isdir(fs_join(target, 'src'))
            assert fs.isfile(fs_join(target, 'src', 'file'))
    fs.rm(target, recursive=True)
    assert not fs.exists(target)
    for loop in range(2):
        fs.put(src + '/', target, recursive=True)
        assert fs.isdir(target)
        assert fs.isfile(fs_join(target, 'file'))
        assert not fs.exists(fs_join(target, 'src'))