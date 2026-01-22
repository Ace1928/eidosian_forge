import os
import re
import sys
import time
from io import BytesIO
from typing import Callable, ClassVar, Dict, Iterator, List, Optional, Tuple
from urllib.parse import parse_qs
from wsgiref.simple_server import (
from dulwich import log_utils
from .protocol import ReceivableProtocol
from .repo import BaseRepo, NotGitRepository, Repo
from .server import (
class ChunkReader:

    def __init__(self, f) -> None:
        self._iter = _chunk_iter(f)
        self._buffer: List[bytes] = []

    def read(self, n):
        while sum(map(len, self._buffer)) < n:
            try:
                self._buffer.append(next(self._iter))
            except StopIteration:
                break
        f = b''.join(self._buffer)
        ret = f[:n]
        self._buffer = [f[n:]]
        return ret