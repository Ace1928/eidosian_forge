import pathlib
import platform
import sys
import pytest
from ..paths import file_uri_to_path, is_relative, normalized_uri
@pytest.mark.skipif(PY35, reason="can't test non-existent paths on py35")
@pytest.mark.skipif(WIN, reason="can't test POSIX paths on Windows")
@pytest.mark.parametrize('root_dir, expected_root_uri', [[str(HOME / 'foo'), (HOME / 'foo').as_uri()]])
def test_normalize_posix_path_home_subdir(root_dir, expected_root_uri):
    assert normalized_uri(root_dir) == expected_root_uri