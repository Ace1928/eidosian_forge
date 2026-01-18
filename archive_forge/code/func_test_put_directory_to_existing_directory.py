from hashlib import md5
from itertools import product
import pytest
from fsspec.tests.abstract.common import GLOB_EDGE_CASES_TESTS
def test_put_directory_to_existing_directory(self, fs, fs_join, fs_target, local_bulk_operations_scenario_0, supports_empty_directories):
    source = local_bulk_operations_scenario_0
    target = fs_target
    fs.mkdir(target)
    if not supports_empty_directories:
        dummy = fs_join(target, 'dummy')
        fs.touch(dummy)
    assert fs.isdir(target)
    for source_slash, target_slash in zip([False, True], [False, True]):
        s = fs_join(source, 'subdir')
        if source_slash:
            s += '/'
        t = target + '/' if target_slash else target
        fs.put(s, t)
        assert fs.ls(target, detail=False) == ([] if supports_empty_directories else [dummy])
        fs.put(s, t, recursive=True)
        if source_slash:
            assert fs.isfile(fs_join(target, 'subfile1'))
            assert fs.isfile(fs_join(target, 'subfile2'))
            assert fs.isdir(fs_join(target, 'nesteddir'))
            assert fs.isfile(fs_join(target, 'nesteddir', 'nestedfile'))
            assert not fs.exists(fs_join(target, 'subdir'))
            fs.rm([fs_join(target, 'subfile1'), fs_join(target, 'subfile2'), fs_join(target, 'nesteddir')], recursive=True)
        else:
            assert fs.isdir(fs_join(target, 'subdir'))
            assert fs.isfile(fs_join(target, 'subdir', 'subfile1'))
            assert fs.isfile(fs_join(target, 'subdir', 'subfile2'))
            assert fs.isdir(fs_join(target, 'subdir', 'nesteddir'))
            assert fs.isfile(fs_join(target, 'subdir', 'nesteddir', 'nestedfile'))
            fs.rm(fs_join(target, 'subdir'), recursive=True)
        assert fs.ls(target, detail=False) == ([] if supports_empty_directories else [dummy])
        fs.put(s, t, recursive=True, maxdepth=1)
        if source_slash:
            assert fs.isfile(fs_join(target, 'subfile1'))
            assert fs.isfile(fs_join(target, 'subfile2'))
            assert not fs.exists(fs_join(target, 'nesteddir'))
            assert not fs.exists(fs_join(target, 'subdir'))
            fs.rm([fs_join(target, 'subfile1'), fs_join(target, 'subfile2')], recursive=True)
        else:
            assert fs.isdir(fs_join(target, 'subdir'))
            assert fs.isfile(fs_join(target, 'subdir', 'subfile1'))
            assert fs.isfile(fs_join(target, 'subdir', 'subfile2'))
            assert not fs.exists(fs_join(target, 'subdir', 'nesteddir'))
            fs.rm(fs_join(target, 'subdir'), recursive=True)
        assert fs.ls(target, detail=False) == ([] if supports_empty_directories else [dummy])