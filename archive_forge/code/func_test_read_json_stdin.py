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
def test_read_json_stdin(monkeypatch):
    input_data = '{\n    "hello": "world"\n}'
    monkeypatch.setattr('sys.stdin', StringIO(input_data))
    data = read_json('-')
    assert len(data) == 1
    assert data['hello'] == 'world'