import pathlib
import platform
import sys
import pytest
from ..paths import file_uri_to_path, is_relative, normalized_uri
@pytest.mark.skipif(not WIN, reason="can't test Windows paths on POSIX")
@pytest.mark.parametrize('root_dir, expected_root_uri', [['c:\\Users\\user1', 'file:///c:/Users/user1'], ['C:\\Users\\user1', 'file:///c:/Users/user1'], ['//VBOXSVR/shared-folder', 'file://vboxsvr/shared-folder/']])
def test_normalize_windows_path_case(root_dir, expected_root_uri):
    try:
        normalized = normalized_uri(root_dir)
    except FileNotFoundError as err:
        if sys.version_info >= (3, 10):
            return
        raise err
    assert normalized == expected_root_uri