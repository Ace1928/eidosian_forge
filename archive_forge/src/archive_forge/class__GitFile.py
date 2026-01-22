import os
import sys
import warnings
from typing import ClassVar, Set
class _GitFile:
    """File that follows the git locking protocol for writes.

    All writes to a file foo will be written into foo.lock in the same
    directory, and the lockfile will be renamed to overwrite the original file
    on close.

    Note: You *must* call close() or abort() on a _GitFile for the lock to be
        released. Typically this will happen in a finally block.
    """
    PROXY_PROPERTIES: ClassVar[Set[str]] = {'closed', 'encoding', 'errors', 'mode', 'name', 'newlines', 'softspace'}
    PROXY_METHODS: ClassVar[Set[str]] = {'__iter__', 'flush', 'fileno', 'isatty', 'read', 'readline', 'readlines', 'seek', 'tell', 'truncate', 'write', 'writelines'}

    def __init__(self, filename, mode, bufsize, mask) -> None:
        self._filename = filename
        if isinstance(self._filename, bytes):
            self._lockfilename = self._filename + b'.lock'
        else:
            self._lockfilename = self._filename + '.lock'
        try:
            fd = os.open(self._lockfilename, os.O_RDWR | os.O_CREAT | os.O_EXCL | getattr(os, 'O_BINARY', 0), mask)
        except FileExistsError as exc:
            raise FileLocked(filename, self._lockfilename) from exc
        self._file = os.fdopen(fd, mode, bufsize)
        self._closed = False
        for method in self.PROXY_METHODS:
            setattr(self, method, getattr(self._file, method))

    def abort(self):
        """Close and discard the lockfile without overwriting the target.

        If the file is already closed, this is a no-op.
        """
        if self._closed:
            return
        self._file.close()
        try:
            os.remove(self._lockfilename)
            self._closed = True
        except FileNotFoundError:
            self._closed = True

    def close(self):
        """Close this file, saving the lockfile over the original.

        Note: If this method fails, it will attempt to delete the lockfile.
            However, it is not guaranteed to do so (e.g. if a filesystem
            becomes suddenly read-only), which will prevent future writes to
            this file until the lockfile is removed manually.

        Raises:
          OSError: if the original file could not be overwritten. The
            lock file is still closed, so further attempts to write to the same
            file object will raise ValueError.
        """
        if self._closed:
            return
        self._file.flush()
        os.fsync(self._file.fileno())
        self._file.close()
        try:
            if getattr(os, 'replace', None) is not None:
                os.replace(self._lockfilename, self._filename)
            elif sys.platform != 'win32':
                os.rename(self._lockfilename, self._filename)
            else:
                _fancy_rename(self._lockfilename, self._filename)
        finally:
            self.abort()

    def __del__(self) -> None:
        if not getattr(self, '_closed', True):
            warnings.warn('unclosed %r' % self, ResourceWarning, stacklevel=2)
            self.abort()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self.abort()
        else:
            self.close()

    def __getattr__(self, name):
        """Proxy property calls to the underlying file."""
        if name in self.PROXY_PROPERTIES:
            return getattr(self._file, name)
        raise AttributeError(name)