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
def test_write_json_stdout(capsys):
    data = {'hello': 'world', 'test': 123}
    expected = ['{\n  "hello":"world",\n  "test":123\n}\n', '{\n  "test":123,\n  "hello":"world"\n}\n']
    write_json('-', data)
    captured = capsys.readouterr()
    assert captured.out in expected