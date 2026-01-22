import fnmatch
import io
import os
import shutil
import tarfile
from typing import Optional, Tuple, Dict, Generator, Union, List
import ray
from ray.util.annotations import DeveloperAPI
from ray.air._internal.filelock import TempFileLock
from ray.air.util.node import _get_node_id_from_node_ip, _force_on_node
def _unpack_dir(stream: io.BytesIO, target_dir: str, *, _retry: bool=True) -> None:
    """Unpack tarfile stream into target directory."""
    stream.seek(0)
    target_dir = os.path.normpath(target_dir)
    try:
        with TempFileLock(f'{target_dir}.lock', timeout=0):
            with tarfile.open(fileobj=stream) as tar:
                tar.extractall(target_dir)
    except TimeoutError:
        with TempFileLock(f'{target_dir}.lock'):
            pass
        if not os.path.exists(target_dir):
            if _retry:
                _unpack_dir(stream, target_dir, _retry=False)
            else:
                raise RuntimeError(f"Target directory {target_dir} does not exist and couldn't be recreated. Please raise an issue on GitHub: https://github.com/ray-project/ray/issues")