import contextlib
import os
import shutil
import stat
import tempfile
from functools import partial
from pathlib import Path
from typing import Callable, Generator, Optional, Union
import yaml
from filelock import BaseFileLock, FileLock
@contextlib.contextmanager
def WeakFileLock(lock_file: Union[str, Path]) -> Generator[BaseFileLock, None, None]:
    lock = FileLock(lock_file)
    lock.acquire()
    yield lock
    try:
        return lock.release()
    except OSError:
        try:
            Path(lock_file).unlink()
        except OSError:
            pass