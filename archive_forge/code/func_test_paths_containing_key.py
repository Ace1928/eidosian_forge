from __future__ import annotations
import os
import pathlib
import site
import stat
import sys
from collections import OrderedDict
from contextlib import contextmanager
import pytest
import yaml
import dask.config
from dask.config import (
def test_paths_containing_key(tmp_path):
    a = {'x': 1, 'y': {'a': 1}}
    b = {'x': 2, 'z': 3, 'y': {'b': 2}}
    pa, pb = (tmp_path / 'a.yaml', tmp_path / 'b.yaml')
    pa.write_text(yaml.dump(a))
    pb.write_text(yaml.dump(b))
    paths = list(paths_containing_key('y.a', paths=[pa, pb]))
    assert paths == [pa]
    assert not list(paths_containing_key('w', paths=[pa, pb]))
    assert not list(paths_containing_key('x', paths=[]))