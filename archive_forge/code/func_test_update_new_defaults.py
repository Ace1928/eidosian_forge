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
def test_update_new_defaults():
    d = {'x': 1, 'y': 1, 'z': {'a': 1, 'b': 1}}
    o = {'x': 1, 'y': 2, 'z': {'a': 1, 'b': 2}, 'c': 2, 'c2': {'d': 2}}
    n = {'x': 3, 'y': 3, 'z': OrderedDict({'a': 3, 'b': 3}), 'c': 3, 'c2': {'d': 3}}
    assert update(o, n, priority='new-defaults', defaults=d) == {'x': 3, 'y': 2, 'z': {'a': 3, 'b': 2}, 'c': 2, 'c2': {'d': 2}}
    assert update(o, n, priority='new-defaults', defaults=o) == update(o, n, priority='new')
    assert update(o, n, priority='new-defaults', defaults=None) == update(o, n, priority='old')