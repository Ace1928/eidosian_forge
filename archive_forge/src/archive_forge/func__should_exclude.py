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
def _should_exclude(candidate: str) -> bool:
    if not exclude:
        return False
    for excl in exclude:
        if fnmatch.fnmatch(candidate, excl):
            return True
    return False