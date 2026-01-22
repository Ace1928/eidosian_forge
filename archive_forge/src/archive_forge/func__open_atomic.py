import os
import contextlib
import json
import shutil
import pathlib
from typing import Any, List
import uuid
from ray.workflow.storage.base import Storage, KeyNotFoundError
import ray.cloudpickle
@contextlib.contextmanager
def _open_atomic(path: pathlib.Path, mode='r'):
    """Open file with atomic file writing support. File reading is also
    adapted to atomic file writing (for example, the backup file
    is used when an atomic write failed previously.)

    TODO(suquark): race condition like two processes writing the
    same file is still not safe. This may not be an issue, because
    in our current implementation, we only need to guarantee the
    file is either fully written or not existing.

    Args:
        path: The file path.
        mode: Open mode same as "open()".

    Returns:
        File object.
    """
    if 'a' in mode or '+' in mode:
        raise ValueError('Atomic open does not support appending.')
    backup_path = path.with_name(f'.{path.name}.backup')
    if 'r' in mode:
        if _file_exists(path):
            f = open(path, mode)
        else:
            raise KeyNotFoundError(path)
        try:
            yield f
        finally:
            f.close()
    elif 'x' in mode:
        if path.exists():
            raise FileExistsError(path)
        tmp_new_fn = path.with_suffix(f'.{path.name}.{uuid.uuid4().hex}')
        if not tmp_new_fn.parent.exists():
            tmp_new_fn.parent.mkdir(parents=True)
        f = open(tmp_new_fn, mode)
        write_ok = True
        try:
            yield f
        except Exception:
            write_ok = False
            raise
        finally:
            f.close()
            if write_ok:
                tmp_new_fn.rename(path)
            else:
                tmp_new_fn.unlink()
    elif 'w' in mode:
        if path.exists():
            if backup_path.exists():
                backup_path.unlink()
            path.rename(backup_path)
        tmp_new_fn = path.with_suffix(f'.{path.name}.{uuid.uuid4().hex}')
        if not tmp_new_fn.parent.exists():
            tmp_new_fn.parent.mkdir(parents=True)
        f = open(tmp_new_fn, mode)
        write_ok = True
        try:
            yield f
        except Exception:
            write_ok = False
            raise
        finally:
            f.close()
            if write_ok:
                tmp_new_fn.rename(path)
                if backup_path.exists():
                    backup_path.unlink()
            else:
                tmp_new_fn.unlink()
    else:
        raise ValueError(f'Unknown file open mode {mode}.')