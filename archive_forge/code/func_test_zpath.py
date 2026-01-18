import os
import pytest
from monty.os import cd, makedirs_p
from monty.os.path import find_exts, zpath
def test_zpath(self):
    fullzpath = zpath(os.path.join(test_dir, 'myfile_gz'))
    assert os.path.join(test_dir, 'myfile_gz.gz') == fullzpath