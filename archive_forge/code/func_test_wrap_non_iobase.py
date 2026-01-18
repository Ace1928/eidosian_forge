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
def test_wrap_non_iobase() -> None:

    class FakeFile:

        def close(self) -> None:
            pass

        def write(self) -> None:
            pass
    wrapped = FakeFile()
    assert not isinstance(wrapped, io.IOBase)
    async_file = trio.wrap_file(wrapped)
    assert isinstance(async_file, AsyncIOWrapper)
    del FakeFile.write
    with pytest.raises(TypeError):
        trio.wrap_file(FakeFile())