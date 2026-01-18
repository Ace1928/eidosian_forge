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
def test_read_jsonl_gzip():
    """Tests reading data from a gzipped .jsonl file."""
    file_contents = [{'hello': 'world'}, {'test': 123}]
    with make_tempdir() as temp_dir:
        file_path = temp_dir / 'tmp.json'
        with gzip.open(file_path, 'w') as f:
            f.writelines([(json_dumps(line) + '\n').encode('utf-8') for line in file_contents])
        assert file_path.exists()
        data = read_gzip_jsonl(file_path)
        assert not hasattr(data, '__len__')
        data = list(data)
    assert len(data) == 2
    assert len(data[0]) == 1
    assert len(data[1]) == 1
    assert data[0]['hello'] == 'world'
    assert data[1]['test'] == 123