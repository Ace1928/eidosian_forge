import errno
import itertools
import logging
import os.path
import tempfile
import traceback
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import (
from pip._internal.utils.misc import enum, rmtree
class TempDirectory:
    """Helper class that owns and cleans up a temporary directory.

    This class can be used as a context manager or as an OO representation of a
    temporary directory.

    Attributes:
        path
            Location to the created temporary directory
        delete
            Whether the directory should be deleted when exiting
            (when used as a contextmanager)

    Methods:
        cleanup()
            Deletes the temporary directory

    When used as a context manager, if the delete attribute is True, on
    exiting the context the temporary directory is deleted.
    """

    def __init__(self, path: Optional[str]=None, delete: Union[bool, None, _Default]=_default, kind: str='temp', globally_managed: bool=False, ignore_cleanup_errors: bool=True):
        super().__init__()
        if delete is _default:
            if path is not None:
                delete = False
            else:
                delete = None
        if path is None:
            path = self._create(kind)
        self._path = path
        self._deleted = False
        self.delete = delete
        self.kind = kind
        self.ignore_cleanup_errors = ignore_cleanup_errors
        if globally_managed:
            assert _tempdir_manager is not None
            _tempdir_manager.enter_context(self)

    @property
    def path(self) -> str:
        assert not self._deleted, f'Attempted to access deleted path: {self._path}'
        return self._path

    def __repr__(self) -> str:
        return f'<{self.__class__.__name__} {self.path!r}>'

    def __enter__(self: _T) -> _T:
        return self

    def __exit__(self, exc: Any, value: Any, tb: Any) -> None:
        if self.delete is not None:
            delete = self.delete
        elif _tempdir_registry:
            delete = _tempdir_registry.get_delete(self.kind)
        else:
            delete = True
        if delete:
            self.cleanup()

    def _create(self, kind: str) -> str:
        """Create a temporary directory and store its path in self.path"""
        path = os.path.realpath(tempfile.mkdtemp(prefix=f'pip-{kind}-'))
        logger.debug('Created temporary directory: %s', path)
        return path

    def cleanup(self) -> None:
        """Remove the temporary directory created and reset state"""
        self._deleted = True
        if not os.path.exists(self._path):
            return
        errors: List[BaseException] = []

        def onerror(func: Callable[..., Any], path: Path, exc_val: BaseException) -> None:
            """Log a warning for a `rmtree` error and continue"""
            formatted_exc = '\n'.join(traceback.format_exception_only(type(exc_val), exc_val))
            formatted_exc = formatted_exc.rstrip()
            if func in (os.unlink, os.remove, os.rmdir):
                logger.debug("Failed to remove a temporary file '%s' due to %s.\n", path, formatted_exc)
            else:
                logger.debug('%s failed with %s.', func.__qualname__, formatted_exc)
            errors.append(exc_val)
        if self.ignore_cleanup_errors:
            try:
                rmtree(self._path, ignore_errors=False)
            except OSError:
                rmtree(self._path, onexc=onerror)
            if errors:
                logger.warning("Failed to remove contents in a temporary directory '%s'.\nYou can safely remove it manually.", self._path)
        else:
            rmtree(self._path)