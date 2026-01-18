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
def test_write_jsonl_file():
    data = [{'hello': 'world'}, {'test': 123}]
    with make_tempdir() as temp_dir:
        file_path = temp_dir / 'tmp.json'
        write_jsonl(file_path, data)
        with Path(file_path).open('r', encoding='utf8') as f:
            assert f.read() == '{"hello":"world"}\n{"test":123}\n'