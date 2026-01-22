from __future__ import annotations
import io
import logging
import os
import threading
import warnings
import weakref
from errno import ESPIPE
from glob import has_magic
from hashlib import sha256
from typing import ClassVar
from .callbacks import DEFAULT_CALLBACK
from .config import apply_config, conf
from .dircache import DirCache
from .transaction import Transaction
from .utils import (
class AbstractBufferedFile(io.IOBase):
    """Convenient class to derive from to provide buffering

    In the case that the backend does not provide a pythonic file-like object
    already, this class contains much of the logic to build one. The only
    methods that need to be overridden are ``_upload_chunk``,
    ``_initiate_upload`` and ``_fetch_range``.
    """
    DEFAULT_BLOCK_SIZE = 5 * 2 ** 20
    _details = None

    def __init__(self, fs, path, mode='rb', block_size='default', autocommit=True, cache_type='readahead', cache_options=None, size=None, **kwargs):
        """
        Template for files with buffered reading and writing

        Parameters
        ----------
        fs: instance of FileSystem
        path: str
            location in file-system
        mode: str
            Normal file modes. Currently only 'wb', 'ab' or 'rb'. Some file
            systems may be read-only, and some may not support append.
        block_size: int
            Buffer size for reading or writing, 'default' for class default
        autocommit: bool
            Whether to write to final destination; may only impact what
            happens when file is being closed.
        cache_type: {"readahead", "none", "mmap", "bytes"}, default "readahead"
            Caching policy in read mode. See the definitions in ``core``.
        cache_options : dict
            Additional options passed to the constructor for the cache specified
            by `cache_type`.
        size: int
            If given and in read mode, suppressed having to look up the file size
        kwargs:
            Gets stored as self.kwargs
        """
        from .core import caches
        self.path = path
        self.fs = fs
        self.mode = mode
        self.blocksize = self.DEFAULT_BLOCK_SIZE if block_size in ['default', None] else block_size
        self.loc = 0
        self.autocommit = autocommit
        self.end = None
        self.start = None
        self.closed = False
        if cache_options is None:
            cache_options = {}
        if 'trim' in kwargs:
            warnings.warn("Passing 'trim' to control the cache behavior has been deprecated. Specify it within the 'cache_options' argument instead.", FutureWarning)
            cache_options['trim'] = kwargs.pop('trim')
        self.kwargs = kwargs
        if mode not in {'ab', 'rb', 'wb'}:
            raise NotImplementedError('File mode not supported')
        if mode == 'rb':
            if size is not None:
                self.size = size
            else:
                self.size = self.details['size']
            self.cache = caches[cache_type](self.blocksize, self._fetch_range, self.size, **cache_options)
        else:
            self.buffer = io.BytesIO()
            self.offset = None
            self.forced = False
            self.location = None

    @property
    def details(self):
        if self._details is None:
            self._details = self.fs.info(self.path)
        return self._details

    @details.setter
    def details(self, value):
        self._details = value
        self.size = value['size']

    @property
    def full_name(self):
        return _unstrip_protocol(self.path, self.fs)

    @property
    def closed(self):
        return getattr(self, '_closed', True)

    @closed.setter
    def closed(self, c):
        self._closed = c

    def __hash__(self):
        if 'w' in self.mode:
            return id(self)
        else:
            return int(tokenize(self.details), 16)

    def __eq__(self, other):
        """Files are equal if they have the same checksum, only in read mode"""
        if self is other:
            return True
        return self.mode == 'rb' and other.mode == 'rb' and (hash(self) == hash(other))

    def commit(self):
        """Move from temp to final destination"""

    def discard(self):
        """Throw away temporary file"""

    def info(self):
        """File information about this path"""
        if 'r' in self.mode:
            return self.details
        else:
            raise ValueError('Info not available while writing')

    def tell(self):
        """Current file location"""
        return self.loc

    def seek(self, loc, whence=0):
        """Set current file location

        Parameters
        ----------
        loc: int
            byte location
        whence: {0, 1, 2}
            from start of file, current location or end of file, resp.
        """
        loc = int(loc)
        if not self.mode == 'rb':
            raise OSError(ESPIPE, 'Seek only available in read mode')
        if whence == 0:
            nloc = loc
        elif whence == 1:
            nloc = self.loc + loc
        elif whence == 2:
            nloc = self.size + loc
        else:
            raise ValueError(f'invalid whence ({whence}, should be 0, 1 or 2)')
        if nloc < 0:
            raise ValueError('Seek before start of file')
        self.loc = nloc
        return self.loc

    def write(self, data):
        """
        Write data to buffer.

        Buffer only sent on flush() or if buffer is greater than
        or equal to blocksize.

        Parameters
        ----------
        data: bytes
            Set of bytes to be written.
        """
        if self.mode not in {'wb', 'ab'}:
            raise ValueError('File not in write mode')
        if self.closed:
            raise ValueError('I/O operation on closed file.')
        if self.forced:
            raise ValueError('This file has been force-flushed, can only close')
        out = self.buffer.write(data)
        self.loc += out
        if self.buffer.tell() >= self.blocksize:
            self.flush()
        return out

    def flush(self, force=False):
        """
        Write buffered data to backend store.

        Writes the current buffer, if it is larger than the block-size, or if
        the file is being closed.

        Parameters
        ----------
        force: bool
            When closing, write the last block even if it is smaller than
            blocks are allowed to be. Disallows further writing to this file.
        """
        if self.closed:
            raise ValueError('Flush on closed file')
        if force and self.forced:
            raise ValueError('Force flush cannot be called more than once')
        if force:
            self.forced = True
        if self.mode not in {'wb', 'ab'}:
            return
        if not force and self.buffer.tell() < self.blocksize:
            return
        if self.offset is None:
            self.offset = 0
            try:
                self._initiate_upload()
            except:
                self.closed = True
                raise
        if self._upload_chunk(final=force) is not False:
            self.offset += self.buffer.seek(0, 2)
            self.buffer = io.BytesIO()

    def _upload_chunk(self, final=False):
        """Write one part of a multi-block file upload

        Parameters
        ==========
        final: bool
            This is the last block, so should complete file, if
            self.autocommit is True.
        """

    def _initiate_upload(self):
        """Create remote file/upload"""
        pass

    def _fetch_range(self, start, end):
        """Get the specified set of bytes from remote"""
        raise NotImplementedError

    def read(self, length=-1):
        """
        Return data from cache, or fetch pieces as necessary

        Parameters
        ----------
        length: int (-1)
            Number of bytes to read; if <0, all remaining bytes.
        """
        length = -1 if length is None else int(length)
        if self.mode != 'rb':
            raise ValueError('File not in read mode')
        if length < 0:
            length = self.size - self.loc
        if self.closed:
            raise ValueError('I/O operation on closed file.')
        logger.debug('%s read: %i - %i', self, self.loc, self.loc + length)
        if length == 0:
            return b''
        out = self.cache._fetch(self.loc, self.loc + length)
        self.loc += len(out)
        return out

    def readinto(self, b):
        """mirrors builtin file's readinto method

        https://docs.python.org/3/library/io.html#io.RawIOBase.readinto
        """
        out = memoryview(b).cast('B')
        data = self.read(out.nbytes)
        out[:len(data)] = data
        return len(data)

    def readuntil(self, char=b'\n', blocks=None):
        """Return data between current position and first occurrence of char

        char is included in the output, except if the end of the tile is
        encountered first.

        Parameters
        ----------
        char: bytes
            Thing to find
        blocks: None or int
            How much to read in each go. Defaults to file blocksize - which may
            mean a new read on every call.
        """
        out = []
        while True:
            start = self.tell()
            part = self.read(blocks or self.blocksize)
            if len(part) == 0:
                break
            found = part.find(char)
            if found > -1:
                out.append(part[:found + len(char)])
                self.seek(start + found + len(char))
                break
            out.append(part)
        return b''.join(out)

    def readline(self):
        """Read until first occurrence of newline character

        Note that, because of character encoding, this is not necessarily a
        true line ending.
        """
        return self.readuntil(b'\n')

    def __next__(self):
        out = self.readline()
        if out:
            return out
        raise StopIteration

    def __iter__(self):
        return self

    def readlines(self):
        """Return all data, split by the newline character"""
        data = self.read()
        lines = data.split(b'\n')
        out = [l + b'\n' for l in lines[:-1]]
        if data.endswith(b'\n'):
            return out
        else:
            return out + [lines[-1]]

    def readinto1(self, b):
        return self.readinto(b)

    def close(self):
        """Close file

        Finalizes writes, discards cache
        """
        if getattr(self, '_unclosable', False):
            return
        if self.closed:
            return
        if self.mode == 'rb':
            self.cache = None
        else:
            if not self.forced:
                self.flush(force=True)
            if self.fs is not None:
                self.fs.invalidate_cache(self.path)
                self.fs.invalidate_cache(self.fs._parent(self.path))
        self.closed = True

    def readable(self):
        """Whether opened for reading"""
        return self.mode == 'rb' and (not self.closed)

    def seekable(self):
        """Whether is seekable (only in read mode)"""
        return self.readable()

    def writable(self):
        """Whether opened for writing"""
        return self.mode in {'wb', 'ab'} and (not self.closed)

    def __del__(self):
        if not self.closed:
            self.close()

    def __str__(self):
        return f'<File-like object {type(self.fs).__name__}, {self.path}>'
    __repr__ = __str__

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()