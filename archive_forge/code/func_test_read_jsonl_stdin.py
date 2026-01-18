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
def test_read_jsonl_stdin(monkeypatch):
    input_data = '{"hello": "world"}\n{"test": 123}'
    monkeypatch.setattr('sys.stdin', StringIO(input_data))
    data = read_jsonl('-')
    assert not hasattr(data, '__len__')
    data = list(data)
    assert len(data) == 2
    assert len(data[0]) == 1
    assert len(data[1]) == 1
    assert data[0]['hello'] == 'world'
    assert data[1]['test'] == 123