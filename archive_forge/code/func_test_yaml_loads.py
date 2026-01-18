from io import StringIO
from pathlib import Path
import pytest
from .._yaml_api import yaml_dumps, yaml_loads, read_yaml, write_yaml
from .._yaml_api import is_yaml_serializable
from ..ruamel_yaml.comments import CommentedMap
from .util import make_tempdir
def test_yaml_loads():
    data = 'a:\n- 1\n- hello\nb:\n  foo: bar\n  baz:\n  - 10.5\n  - 120\n'
    result = yaml_loads(data)
    assert not isinstance(result, CommentedMap)
    assert result == {'a': [1, 'hello'], 'b': {'foo': 'bar', 'baz': [10.5, 120]}}