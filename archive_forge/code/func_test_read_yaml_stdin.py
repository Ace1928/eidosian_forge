from io import StringIO
from pathlib import Path
import pytest
from .._yaml_api import yaml_dumps, yaml_loads, read_yaml, write_yaml
from .._yaml_api import is_yaml_serializable
from ..ruamel_yaml.comments import CommentedMap
from .util import make_tempdir
def test_read_yaml_stdin(monkeypatch):
    input_data = 'a:\n  - 1\n  - hello\nb:\n  foo: bar\n  baz:\n    - 10.5\n    - 120\n'
    monkeypatch.setattr('sys.stdin', StringIO(input_data))
    data = read_yaml('-')
    assert len(data) == 2
    assert data['a'] == [1, 'hello']