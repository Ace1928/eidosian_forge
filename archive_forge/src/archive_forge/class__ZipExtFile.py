from __future__ import print_function, unicode_literals
import sys
import typing
import six
import zipfile
from datetime import datetime
from . import errors
from ._url_tools import url_quote
from .base import FS
from .compress import write_zip
from .enums import ResourceType, Seek
from .info import Info
from .iotools import RawWrapper
from .memoryfs import MemoryFS
from .opener import open_fs
from .path import dirname, forcedir, normpath, relpath
from .permissions import Permissions
from .time import datetime_to_epoch
from .wrapfs import WrapFS
class _ZipExtFile(RawWrapper):

    def __init__(self, fs, name):
        self._zip = _zip = fs._zip
        self._end = _zip.getinfo(name).file_size
        self._pos = 0
        super(_ZipExtFile, self).__init__(_zip.open(name), 'r', name)
    if sys.version_info < (3, 7):

        def read(self, size=-1):
            buf = self._f.read(-1 if size is None else size)
            self._pos += len(buf)
            return buf

        def read1(self, size=-1):
            buf = self._f.read1(-1 if size is None else size)
            self._pos += len(buf)
            return buf

        def tell(self):
            return self._pos

        def seekable(self):
            return True

        def seek(self, offset, whence=Seek.set):
            """Change stream position.

            Change the stream position to the given byte offset. The
            offset is interpreted relative to the position indicated by
            ``whence``.

            Arguments:
                offset (int): the offset to the new position, in bytes.
                whence (int): the position reference. Possible values are:
                    * `Seek.set`: start of stream (the default).
                    * `Seek.current`: current position; offset may be negative.
                    * `Seek.end`: end of stream; offset must be negative.

            Returns:
                int: the new absolute position.

            Raises:
                ValueError: when ``whence`` is not known, or ``offset``
                    is invalid.

            Note:
                Zip compression does not support seeking, so the seeking
                is emulated. Seeking somewhere else than the current position
                will need to either:
                    * reopen the file and restart decompression
                    * read and discard data to advance in the file

            """
            _whence = int(whence)
            if _whence == Seek.current:
                offset += self._pos
            if _whence == Seek.current or _whence == Seek.set:
                if offset < 0:
                    raise ValueError('Negative seek position {}'.format(offset))
            elif _whence == Seek.end:
                if offset > 0:
                    raise ValueError('Positive seek position {}'.format(offset))
                offset += self._end
            else:
                raise ValueError('Invalid whence ({}, should be {}, {} or {})'.format(_whence, Seek.set, Seek.current, Seek.end))
            if offset < self._pos:
                self._f = self._zip.open(self.name)
                self._pos = 0
            self.read(offset - self._pos)
            return self._pos
    else:

        def seek(self, offset, whence=Seek.set):
            """Change stream position.

            Change the stream position to the given byte offset. The
            offset is interpreted relative to the position indicated by
            ``whence``.

            Arguments:
                offset (int): the offset to the new position, in bytes.
                whence (int): the position reference. Possible values are:
                    * `Seek.set`: start of stream (the default).
                    * `Seek.current`: current position; offset may be negative.
                    * `Seek.end`: end of stream; offset must be negative.

            Returns:
                int: the new absolute position.

            Raises:
                ValueError: when ``whence`` is not known, or ``offset``
                    is invalid.

            """
            _whence = int(whence)
            _pos = self.tell()
            if _whence == Seek.set:
                if offset < 0:
                    raise ValueError('Negative seek position {}'.format(offset))
            elif _whence == Seek.current:
                if _pos + offset < 0:
                    raise ValueError('Negative seek position {}'.format(offset))
            elif _whence == Seek.end:
                if offset > 0:
                    raise ValueError('Positive seek position {}'.format(offset))
            else:
                raise ValueError('Invalid whence ({}, should be {}, {} or {})'.format(_whence, Seek.set, Seek.current, Seek.end))
            return self._f.seek(offset, _whence)