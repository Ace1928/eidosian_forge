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
def _sync_dir_on_same_node(ip: str, source_path: str, target_path: str, exclude: Optional[List]=None, return_futures: bool=False) -> Optional[ray.ObjectRef]:
    """Synchronize directory to another directory on the same node.

    Per default, this function will collect information about already existing
    files in the target directory. All files will be copied over.

    Args:
        ip: IP of the node.
        source_path: Path to source directory.
        target_path: Path to target directory.
        exclude: Pattern of files to exclude, e.g.
            ``["*/checkpoint_*]`` to exclude trial checkpoints.
        return_futures: If True, returns a future of the copy task.

    Returns:
        None, or future of the copy task.

    """
    node_id = _get_node_id_from_node_ip(ip)
    copy_on_node = _remote_copy_dir.options(num_cpus=0, **_force_on_node(node_id))
    copy_future = copy_on_node.remote(source_dir=source_path, target_dir=target_path, exclude=exclude)
    if return_futures:
        return copy_future
    return ray.get(copy_future)