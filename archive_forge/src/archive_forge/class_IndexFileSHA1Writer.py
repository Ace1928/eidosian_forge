from abc import abstractmethod
import contextlib
from functools import wraps
import getpass
import logging
import os
import os.path as osp
import pathlib
import platform
import re
import shutil
import stat
import subprocess
import sys
import time
from urllib.parse import urlsplit, urlunsplit
import warnings
from typing import (
from .types import (
from gitdb.util import (  # noqa: F401  # @IgnorePep8
class IndexFileSHA1Writer:
    """Wrapper around a file-like object that remembers the SHA1 of
    the data written to it. It will write a sha when the stream is closed
    or if the asked for explicitly using write_sha.

    Only useful to the index file.

    :note: Based on the dulwich project.
    """
    __slots__ = ('f', 'sha1')

    def __init__(self, f: IO) -> None:
        self.f = f
        self.sha1 = make_sha(b'')

    def write(self, data: AnyStr) -> int:
        self.sha1.update(data)
        return self.f.write(data)

    def write_sha(self) -> bytes:
        sha = self.sha1.digest()
        self.f.write(sha)
        return sha

    def close(self) -> bytes:
        sha = self.write_sha()
        self.f.close()
        return sha

    def tell(self) -> int:
        return self.f.tell()