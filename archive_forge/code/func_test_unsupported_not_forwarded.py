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
def test_unsupported_not_forwarded() -> None:

    class FakeFile(io.RawIOBase):

        def unsupported_attr(self) -> None:
            pass
    async_file = trio.wrap_file(FakeFile())
    assert hasattr(async_file.wrapped, 'unsupported_attr')
    with pytest.raises(AttributeError):
        async_file.unsupported_attr