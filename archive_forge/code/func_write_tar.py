from __future__ import print_function, unicode_literals
import typing
from typing import IO, cast
import os
import six
import tarfile
from collections import OrderedDict
from . import errors
from ._url_tools import url_quote
from .base import FS
from .compress import write_tar
from .enums import ResourceType
from .errors import IllegalBackReference, NoURL
from .info import Info
from .iotools import RawWrapper
from .opener import open_fs
from .path import basename, frombase, isbase, normpath, parts, relpath
from .permissions import Permissions
from .wrapfs import WrapFS
def write_tar(self, file=None, compression=None, encoding=None):
    """Write tar to a file.

        Arguments:
            file (str or io.IOBase, optional): Destination file, may be
                a file name or an open file object.
            compression (str, optional): Compression to use (one of
                the constants defined in `tarfile` in the stdlib).
            encoding (str, optional): The character encoding to use
                (default uses the encoding defined in
                `~WriteTarFS.__init__`).

        Note:
            This is called automatically when the TarFS is closed.

        """
    if not self.isclosed():
        write_tar(self._temp_fs, file or self._file, compression=compression or self.compression, encoding=encoding or self.encoding)