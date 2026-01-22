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
def _get_recursive_files_and_stats(path: str) -> Dict[str, Tuple[float, int]]:
    """Return dict of files mapping to stats in ``path``.

    This function scans a directory ``path`` recursively and returns a dict
    mapping each contained file to a tuple of (mtime, filesize).

    mtime and filesize are returned from ``os.lstat`` and are usually a
    floating point number (timestamp) and an int (filesize in bytes).
    """
    files_stats = {}
    for root, dirs, files in os.walk(path, topdown=False):
        rel_root = os.path.relpath(root, path)
        for file in files:
            try:
                key = os.path.join(rel_root, file)
                stat = os.lstat(os.path.join(path, key))
                files_stats[key] = (stat.st_mtime, stat.st_size)
            except FileNotFoundError:
                pass
    return files_stats