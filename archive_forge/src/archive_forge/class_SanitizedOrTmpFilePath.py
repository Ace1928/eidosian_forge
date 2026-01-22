import contextlib
import os
import platform
import re
import shutil
import tempfile
from typing import Any, Iterator, List, Mapping, Optional, Tuple, Union
from cmdstanpy import _TMPDIR
from .json import write_stan_json
from .logging import get_logger
class SanitizedOrTmpFilePath:
    """
    Context manager for tmpfiles, handles special characters in filepath.
    """
    UNIXISH_PATTERN = re.compile('[\\s~]')
    WINDOWS_PATTERN = re.compile('\\s')

    @classmethod
    def _has_special_chars(cls, file_path: str) -> bool:
        if platform.system() == 'Windows':
            return bool(cls.WINDOWS_PATTERN.search(file_path))
        return bool(cls.UNIXISH_PATTERN.search(file_path))

    def __init__(self, file_path: str):
        self._tmpdir = None
        if self._has_special_chars(os.path.abspath(file_path)):
            base_path, file_name = os.path.split(os.path.abspath(file_path))
            os.makedirs(base_path, exist_ok=True)
            try:
                short_base_path = windows_short_path(base_path)
                if os.path.exists(short_base_path):
                    file_path = os.path.join(short_base_path, file_name)
            except RuntimeError:
                pass
        if self._has_special_chars(os.path.abspath(file_path)):
            tmpdir = tempfile.mkdtemp()
            if self._has_special_chars(tmpdir):
                raise RuntimeError('Unable to generate temporary path without spaces or special characters! \n Please move your stan file to a location without spaces or special characters.')
            _, path = tempfile.mkstemp(suffix='.stan', dir=tmpdir)
            shutil.copy(file_path, path)
            self._path = path
            self._tmpdir = tmpdir
        else:
            self._path = file_path

    def __enter__(self) -> Tuple[str, bool]:
        return (self._path, self._tmpdir is not None)

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._tmpdir:
            shutil.rmtree(self._tmpdir, ignore_errors=True)