import abc
import asyncio
import base64
import hashlib
import inspect
import keyword
import os
import re
import warnings
from contextlib import contextmanager
from functools import wraps
from pathlib import Path
from types import MappingProxyType
from typing import (
from yarl import URL, __version__ as yarl_version  # type: ignore[attr-defined]
from . import hdrs
from .abc import AbstractMatchInfo, AbstractRouter, AbstractView
from .helpers import DEBUG
from .http import HttpVersion11
from .typedefs import Handler, PathLike
from .web_exceptions import (
from .web_fileresponse import FileResponse
from .web_request import Request
from .web_response import Response, StreamResponse
from .web_routedef import AbstractRouteDef
def add_static(self, prefix: str, path: PathLike, *, name: Optional[str]=None, expect_handler: Optional[_ExpectHandler]=None, chunk_size: int=256 * 1024, show_index: bool=False, follow_symlinks: bool=False, append_version: bool=False) -> AbstractResource:
    """Add static files view.

        prefix - url prefix
        path - folder with files

        """
    assert prefix.startswith('/')
    if prefix.endswith('/'):
        prefix = prefix[:-1]
    resource = StaticResource(prefix, path, name=name, expect_handler=expect_handler, chunk_size=chunk_size, show_index=show_index, follow_symlinks=follow_symlinks, append_version=append_version)
    self.register_resource(resource)
    return resource