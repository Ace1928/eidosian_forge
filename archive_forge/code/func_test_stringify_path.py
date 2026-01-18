from __future__ import annotations
import io
import os
import pathlib
import pytest
from fsspec.utils import (
def test_stringify_path():
    test_filepath = os.path.join('path', 'to', 'file.txt')
    path = pathlib.Path(test_filepath)
    assert stringify_path(path) == test_filepath

    class CustomFSPath:
        """For testing fspath on unknown objects"""

        def __init__(self, path):
            self.path = path

        def __fspath__(self):
            return self.path
    path = CustomFSPath(test_filepath)
    assert stringify_path(path) == test_filepath
    path = (1, 2, 3)
    assert stringify_path(path) is path