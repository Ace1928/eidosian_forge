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
def test_update_list_to_dict():
    a = {'x': {'y': 1, 'z': 2}}
    b = {'x': [1, 2], 'w': 3}
    update(b, a)
    assert b == {'x': {'y': 1, 'z': 2}, 'w': 3}