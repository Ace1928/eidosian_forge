import pytest
from io import StringIO
from pathlib import Path
import gzip
import numpy
from .._json_api import (
from .._json_api import write_gzip_json, json_dumps, is_json_serializable
from .._json_api import json_loads
from ..util import force_string
from .util import make_tempdir
def test_write_json_file_gzip():
    data = {'hello': 'world', 'test': 123}
    expected = ['{\n  "hello":"world",\n  "test":123\n}', '{\n  "test":123,\n  "hello":"world"\n}']
    with make_tempdir() as temp_dir:
        file_path = force_string(temp_dir / 'tmp.json')
        write_gzip_json(file_path, data)
        with gzip.open(file_path, 'r') as f:
            assert f.read().decode('utf8') in expected