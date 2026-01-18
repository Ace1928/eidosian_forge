import os
import unittest
import pytest
from monty.io import (
def test_zopen(self):
    with zopen(os.path.join(test_dir, 'myfile_gz.gz'), mode='rt') as f:
        assert f.read() == 'HelloWorld.\n\n'
    with zopen(os.path.join(test_dir, 'myfile_bz2.bz2'), mode='rt') as f:
        assert f.read() == 'HelloWorld.\n\n'
    with zopen(os.path.join(test_dir, 'myfile_bz2.bz2'), 'rt') as f:
        assert f.read() == 'HelloWorld.\n\n'
    with zopen(os.path.join(test_dir, 'myfile_xz.xz'), 'rt') as f:
        assert f.read() == 'HelloWorld.\n\n'
    with zopen(os.path.join(test_dir, 'myfile_lzma.lzma'), 'rt') as f:
        assert f.read() == 'HelloWorld.\n\n'
    with zopen(os.path.join(test_dir, 'myfile'), mode='rt') as f:
        assert f.read() == 'HelloWorld.\n\n'