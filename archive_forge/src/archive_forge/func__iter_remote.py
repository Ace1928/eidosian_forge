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
def _iter_remote(actor: ray.ActorID) -> Generator[bytes, None, None]:
    """Iterate over actor task and return as generator."""
    while True:
        buffer = ray.get(actor.next.remote())
        if buffer is None:
            return
        yield buffer