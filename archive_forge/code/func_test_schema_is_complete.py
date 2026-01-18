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
def test_schema_is_complete():
    root_dir = pathlib.Path(__file__).parent.parent
    config = yaml.safe_load((root_dir / 'dask.yaml').read_text())
    schema = yaml.safe_load((root_dir / 'dask-schema.yaml').read_text())

    def test_matches(c, s):
        for k, v in c.items():
            if list(c) != list(s['properties']):
                raise ValueError("\nThe dask.yaml and dask-schema.yaml files are not in sync.\nThis usually happens when we add a new configuration value,\nbut don't add the schema of that value to the dask-schema.yaml file\nPlease modify these files to include the missing values: \n\n    dask.yaml:        {}\n    dask-schema.yaml: {}\n\nExamples in these files should be a good start, \neven if you are not familiar with the jsonschema spec".format(sorted(c), sorted(s['properties'])))
            if isinstance(v, dict):
                test_matches(c[k], s['properties'][k])
    test_matches(config, schema)