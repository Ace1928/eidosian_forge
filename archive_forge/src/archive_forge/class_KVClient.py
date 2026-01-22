import os
import re
import urllib
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional
from ray._private.client_mode_hook import client_mode_hook
from ray._private.utils import _add_creatable_buckets_param_if_s3_uri, load_class
from ray._private.auto_init_hook import wrap_auto_init
class KVClient:
    """Simple KV API built on the underlying filesystem.

    This is a convenience wrapper around get_filesystem() and working with files.
    Slashes in the path are interpreted as directory delimiters.
    """

    def __init__(self, fs: 'pyarrow.fs.FileSystem', prefix: str):
        """Use storage.get_client() to construct KVClient."""
        self.fs = fs
        self.root = Path(prefix)

    def put(self, path: str, value: bytes) -> None:
        """Save a blob in persistent storage at the given path, if possible.

        Examples:
            .. testcode::

                import ray
                from ray._private import storage

                ray.shutdown()

                ray.init(storage="/tmp/storage/cluster_1/storage")
                client = storage.get_client("my_app")
                client.put("path/foo.txt", b"bar")

        Args:
            path: Relative directory of the blobs.
            value: String value to save.
        """
        full_path = self._resolve_path(path)
        parent_dir = os.path.dirname(full_path)
        try:
            with self.fs.open_output_stream(full_path) as f:
                f.write(value)
        except FileNotFoundError:
            self.fs.create_dir(parent_dir)
            with self.fs.open_output_stream(full_path) as f:
                f.write(value)

    def get(self, path: str) -> bytes:
        """Load a blob from persistent storage at the given path, if possible.

        Examples:
            .. testcode::

                import ray
                from ray._private import storage

                ray.shutdown()

                ray.init(storage="/tmp/storage/cluster_1/storage")

                client = storage.get_client("my_app")
                client.put("path/foo.txt", b"bar")
                assert client.get("path/foo.txt") == b"bar"
                assert client.get("invalid") is None

        Args:
            path: Relative directory of the blobs.

        Returns:
            String content of the blob, or None if not found.
        """
        full_path = self._resolve_path(path)
        try:
            with self.fs.open_input_stream(full_path) as f:
                return f.read()
        except FileNotFoundError:
            return None
        except OSError as e:
            if _is_os_error_file_not_found(e):
                return None
            raise e

    def delete(self, path: str) -> bool:
        """Load the blob from persistent storage at the given path, if possible.

        Examples:
            .. testcode::

                import ray
                from ray._private import storage

                ray.shutdown()

                ray.init(storage="/tmp/storage/cluster_1/storage")

                client = storage.get_client("my_app")
                client.put("path/foo.txt", b"bar")
                assert client.delete("path/foo.txt")

        Args:
            path: Relative directory of the blob.

        Returns:
            Whether the blob was deleted.
        """
        full_path = self._resolve_path(path)
        try:
            self.fs.delete_file(full_path)
            return True
        except FileNotFoundError:
            return False
        except OSError as e:
            if _is_os_error_file_not_found(e):
                return False
            raise e

    def delete_dir(self, path: str) -> bool:
        """Delete a directory and its contents, recursively.

        Examples:
            .. testcode::

                import ray
                from ray._private import storage

                ray.shutdown()

                ray.init(storage="/tmp/storage/cluster_1/storage")

                client = storage.get_client("my_app")
                client.put("path/foo.txt", b"bar")
                assert client.delete_dir("path")

        Args:
            path: Relative directory of the blob.

        Returns:
            Whether the dir was deleted.
        """
        full_path = self._resolve_path(path)
        try:
            self.fs.delete_dir(full_path)
            return True
        except FileNotFoundError:
            return False
        except OSError as e:
            if _is_os_error_file_not_found(e):
                return False
            raise e

    def get_info(self, path: str) -> Optional['pyarrow.fs.FileInfo']:
        """Get info about the persistent blob at the given path, if possible.

        Examples:
            .. testcode::

                import ray
                from ray._private import storage

                ray.shutdown()

                ray.init(storage="/tmp/storage/cluster_1/storage")

                client = storage.get_client("my_app")
                client.put("path/foo.txt", b"bar")

                print(client.get_info("path/foo.txt"))

                print(client.get_info("path/does_not_exist.txt"))

            .. testoutput::

                <FileInfo for '.../my_app/path/foo.txt': type=FileType.File, size=3>
                None

        Args:
            path: Relative directory of the blob.

        Returns:
            Info about the blob, or None if it doesn't exist.
        """
        import pyarrow.fs
        full_path = self._resolve_path(path)
        info = self.fs.get_file_info([full_path])[0]
        if info.type == pyarrow.fs.FileType.NotFound:
            return None
        return info

    def list(self, path: str) -> List['pyarrow.fs.FileInfo']:
        """List blobs and sub-dirs in the given path, if possible.

        Examples:

            >>> import ray
            >>> from ray._private import storage
            >>> ray.shutdown()

            Normal usage.

            >>> ray.init(storage="/tmp/storage/cluster_1/storage")
            RayContext(...)
            >>> client = storage.get_client("my_app")
            >>> client.put("path/foo.txt", b"bar")
            >>> client.list("path")
            [<FileInfo for '.../my_app/path/foo.txt': type=FileType.File, size=3>]

            Non-existent path.

            >>> client.list("does_not_exist")
            Traceback (most recent call last):
                ...
            FileNotFoundError: ... No such file or directory

            Not a directory.

            >>> client.list("path/foo.txt")
            Traceback (most recent call last):
                ...
            NotADirectoryError: ... Not a directory

        Args:
            path: Relative directory to list from.

        Returns:
            List of file-info objects for the directory contents.

        Raises:
            FileNotFoundError if the given path is not found.
            NotADirectoryError if the given path isn't a valid directory.
        """
        from pyarrow.fs import FileSelector, FileType, LocalFileSystem
        full_path = self._resolve_path(path)
        selector = FileSelector(full_path, recursive=False)
        try:
            files = self.fs.get_file_info(selector)
        except FileNotFoundError as e:
            raise e
        except OSError as e:
            if _is_os_error_file_not_found(e):
                raise FileNotFoundError(*e.args)
            raise e
        if self.fs is not LocalFileSystem and (not files):
            info = self.fs.get_file_info([full_path])[0]
            if info.type == FileType.File:
                raise NotADirectoryError(f"Cannot list directory '{full_path}'. Detail: [errno 20] Not a directory")
        return files

    def _resolve_path(self, path: str) -> str:
        from pyarrow.fs import LocalFileSystem
        if isinstance(self.fs, LocalFileSystem):
            joined = self.root.joinpath(path).resolve()
            joined.relative_to(self.root.resolve())
            return str(joined)

        def _normalize_path(p: str) -> str:
            segments = []
            for s in p.replace('\\', '/').split('/'):
                if s == '..':
                    if not segments:
                        raise ValueError('Path goes beyond root.')
                    segments.pop()
                elif s not in ('.', ''):
                    segments.append(s)
            return '/'.join(segments)
        root = _normalize_path(str(self.root))
        joined = _normalize_path(str(self.root.joinpath(path)))
        if not joined.startswith(root):
            raise ValueError(f'{joined!r} does not start with {root!r}')
        return joined