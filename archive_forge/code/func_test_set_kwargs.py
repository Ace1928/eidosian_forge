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
def test_set_kwargs():
    with dask.config.set(foo__bar=1, foo__baz=2):
        assert config['foo'] == {'bar': 1, 'baz': 2}
    assert 'foo' not in config
    with dask.config.set({'foo.bar': 1, 'foo.baz': 2}, foo__buzz=3, foo__bar=4):
        assert config['foo'] == {'bar': 4, 'baz': 2, 'buzz': 3}
    assert 'foo' not in config
    with dask.config.set({'foo': {'bar': 1, 'baz': 2}}, foo__buzz=3, foo__bar=4):
        assert config['foo'] == {'bar': 4, 'baz': 2, 'buzz': 3}
    assert 'foo' not in config