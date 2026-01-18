from __future__ import annotations
import importlib
import io
import os
import re
from typing import TYPE_CHECKING
from unittest import mock
from unittest.mock import sentinel
import pytest
import trio
from trio import _core, _file_io
from trio._file_io import _FILE_ASYNC_METHODS, _FILE_SYNC_ATTRS, AsyncIOWrapper
def test_type_stubs_match_lists() -> None:
    """Check the manual stubs match the list of wrapped methods."""
    assert _file_io.__spec__ is not None
    loader = _file_io.__spec__.loader
    assert isinstance(loader, importlib.abc.SourceLoader)
    source = io.StringIO(loader.get_source('trio._file_io'))
    for line in source:
        if 'class AsyncIOWrapper' in line:
            break
    else:
        pytest.fail('No class definition line?')
    for line in source:
        if 'if TYPE_CHECKING' in line:
            break
    else:
        pytest.fail('No TYPE CHECKING line?')
    found: list[tuple[str, str]] = []
    for line in source:
        if line.strip() and (not line.startswith(' ' * 8)):
            break
        match = re.match('\\s*(async )?def ([a-zA-Z0-9_]+)\\(', line)
        if match is not None:
            kind = 'async' if match.group(1) is not None else 'sync'
            found.append((match.group(2), kind))
    expected = [(fname, 'async') for fname in _FILE_ASYNC_METHODS]
    expected += [(fname, 'sync') for fname in _FILE_SYNC_ATTRS]
    found.sort()
    expected.sort()
    assert found == expected