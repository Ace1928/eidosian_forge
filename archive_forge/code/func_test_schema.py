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
def test_schema():
    jsonschema = pytest.importorskip('jsonschema')
    root_dir = pathlib.Path(__file__).parent.parent
    config = yaml.safe_load((root_dir / 'dask.yaml').read_text())
    schema = yaml.safe_load((root_dir / 'dask-schema.yaml').read_text())
    jsonschema.validate(config, schema)