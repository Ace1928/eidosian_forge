import dataclasses
import glob as py_glob
import io
import os
import os.path
import sys
import tempfile
from tensorboard.compat.tensorflow_stub import compat, errors
class FSSpecFileSystem:
    """Provides filesystem access via fsspec.

    The current gfile interface doesn't map perfectly to the fsspec interface
    leading to some notable inefficiencies.

    * Reads and writes to files cause the file to be reopened each time which
      can cause a performance hit when accessing local file systems.
    * walk doesn't use the native fsspec walk function so performance may be
      slower.

    See https://github.com/tensorflow/tensorboard/issues/5286 for more info on
    limitations.
    """
    SEPARATOR = '://'
    CHAIN_SEPARATOR = '::'

    def _validate_path(self, path):
        parts = path.split(self.CHAIN_SEPARATOR)
        for part in parts[:-1]:
            if self.SEPARATOR in part:
                raise errors.InvalidArgumentError(None, None, 'fsspec URL must only have paths in the last chained filesystem, got {}'.format(path))

    def _translate_errors(func):

        def func_wrapper(self, *args, **kwargs):
            try:
                return func(self, *args, **kwargs)
            except FileNotFoundError as e:
                raise errors.NotFoundError(None, None, str(e))
        return func_wrapper

    def _fs_path(self, filename):
        if isinstance(filename, bytes):
            filename = filename.decode('utf-8')
        self._validate_path(filename)
        fs, path = fsspec.core.url_to_fs(filename)
        return (fs, path)

    @_translate_errors
    def exists(self, filename):
        """Determines whether a path exists or not."""
        fs, path = self._fs_path(filename)
        return fs.exists(path)

    def _join(self, sep, paths):
        """
        _join joins the paths with the given separator.
        """
        result = []
        for part in paths:
            if part.startswith(sep):
                result = []
            if result and result[-1] and (not result[-1].endswith(sep)):
                result.append(sep)
            result.append(part)
        return ''.join(result)

    @_translate_errors
    def join(self, path, *paths):
        """Join paths with a slash."""
        self._validate_path(path)
        before, sep, last_path = path.rpartition(self.CHAIN_SEPARATOR)
        chain_prefix = before + sep
        protocol, path = fsspec.core.split_protocol(last_path)
        fs = fsspec.get_filesystem_class(protocol)
        if protocol:
            chain_prefix += protocol + self.SEPARATOR
        return chain_prefix + self._join(fs.sep, (path,) + paths)

    @_translate_errors
    def read(self, filename, binary_mode=False, size=None, continue_from=None):
        """Reads contents of a file to a string.

        Args:
            filename: string, a path
            binary_mode: bool, read as binary if True, otherwise text
            size: int, number of bytes or characters to read, otherwise
                read all the contents of the file (from the continuation
                marker, if present).
            continue_from: An opaque value returned from a prior invocation of
                `read(...)` marking the last read position, so that reading
                may continue from there.  Otherwise read from the beginning.

        Returns:
            A tuple of `(data, continuation_token)` where `data' provides either
            bytes read from the file (if `binary_mode == true`) or the decoded
            string representation thereof (otherwise), and `continuation_token`
            is an opaque value that can be passed to the next invocation of
            `read(...) ' in order to continue from the last read position.
        """
        fs, path = self._fs_path(filename)
        mode = 'rb' if binary_mode else 'r'
        encoding = None if binary_mode else 'utf8'
        if not exists(filename):
            raise errors.NotFoundError(None, None, 'Not Found: ' + compat.as_text(filename))
        with fs.open(path, mode, encoding=encoding) as f:
            if continue_from is not None:
                if not f.seekable():
                    raise errors.InvalidArgumentError(None, None, '{} is not seekable'.format(filename))
                offset = continue_from.get('opaque_offset', None)
                if offset is not None:
                    f.seek(offset)
            data = f.read(size)
            continuation_token = {'opaque_offset': f.tell()} if f.seekable() else {}
            return (data, continuation_token)

    @_translate_errors
    def write(self, filename, file_content, binary_mode=False):
        """Writes string file contents to a file.

        Args:
            filename: string, a path
            file_content: string, the contents
            binary_mode: bool, write as binary if True, otherwise text
        """
        self._write(filename, file_content, 'wb' if binary_mode else 'w')

    @_translate_errors
    def append(self, filename, file_content, binary_mode=False):
        """Append string file contents to a file.

        Args:
            filename: string, a path
            file_content: string, the contents to append
            binary_mode: bool, write as binary if True, otherwise text
        """
        self._write(filename, file_content, 'ab' if binary_mode else 'a')

    def _write(self, filename, file_content, mode):
        fs, path = self._fs_path(filename)
        encoding = None if 'b' in mode else 'utf8'
        with fs.open(path, mode, encoding=encoding) as f:
            compatify = compat.as_bytes if 'b' in mode else compat.as_text
            f.write(compatify(file_content))

    def _get_chain_protocol_prefix(self, filename):
        chain_prefix, chain_sep, last_path = filename.rpartition(self.CHAIN_SEPARATOR)
        protocol, sep, _ = last_path.rpartition(self.SEPARATOR)
        return chain_prefix + chain_sep + protocol + sep

    @_translate_errors
    def glob(self, filename):
        """Returns a list of files that match the given pattern(s)."""
        if isinstance(filename, bytes):
            filename = filename.decode('utf-8')
        fs, path = self._fs_path(filename)
        files = fs.glob(path)
        if self.SEPARATOR not in filename and self.CHAIN_SEPARATOR not in filename:
            return files
        prefix = self._get_chain_protocol_prefix(filename)
        return [file if self.SEPARATOR in file or self.CHAIN_SEPARATOR in file else prefix + file for file in files]

    @_translate_errors
    def isdir(self, dirname):
        """Returns whether the path is a directory or not."""
        fs, path = self._fs_path(dirname)
        return fs.isdir(path)

    @_translate_errors
    def listdir(self, dirname):
        """Returns a list of entries contained within a directory."""
        fs, path = self._fs_path(dirname)
        files = fs.listdir(path, detail=False)
        files = [os.path.basename(fname) for fname in files]
        return files

    @_translate_errors
    def makedirs(self, dirname):
        """Creates a directory and all parent/intermediate directories."""
        fs, path = self._fs_path(dirname)
        return fs.makedirs(path, exist_ok=True)

    @_translate_errors
    def stat(self, filename):
        """Returns file statistics for a given path."""
        fs, path = self._fs_path(filename)
        return StatData(fs.size(path))