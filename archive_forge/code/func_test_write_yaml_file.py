from io import StringIO
from pathlib import Path
import pytest
from .._yaml_api import yaml_dumps, yaml_loads, read_yaml, write_yaml
from .._yaml_api import is_yaml_serializable
from ..ruamel_yaml.comments import CommentedMap
from .util import make_tempdir
def test_write_yaml_file():
    data = {'hello': 'world', 'test': [123, 456]}
    expected = 'hello: world\ntest:\n  - 123\n  - 456\n'
    with make_tempdir() as temp_dir:
        file_path = temp_dir / 'tmp.yaml'
        write_yaml(file_path, data)
        with Path(file_path).open('r', encoding='utf8') as f:
            assert f.read() == expected