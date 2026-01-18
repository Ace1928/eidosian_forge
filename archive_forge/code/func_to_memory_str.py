import json
import hashlib
import datetime
from typing import Any, Dict, List, Union, Optional
from collections import OrderedDict
from libcloud.compute.base import Node, NodeSize, NodeImage
from libcloud.compute.types import NodeState
from libcloud.container.base import Container, ContainerImage, ContainerDriver, ContainerCluster
from libcloud.container.types import ContainerState
from libcloud.common.exceptions import BaseHTTPError
from libcloud.common.kubernetes import (
from libcloud.container.providers import Provider
def to_memory_str(n_bytes: int, unit: Optional[str]=None) -> str:
    """Convert number of bytes to k8s memory string
    (e.g. 1293942784 -> '1234Mi')
    """
    if n_bytes == 0:
        return '0K'
    n_bytes = int(n_bytes)
    memory_str = None
    if unit is None:
        for unit, multiplier in reversed(K8S_UNIT_MAP.items()):
            converted_n_bytes_float = n_bytes / multiplier
            converted_n_bytes = n_bytes // multiplier
            memory_str = f'{converted_n_bytes}{unit}'
            if converted_n_bytes_float % 1 == 0:
                break
    elif K8S_UNIT_MAP.get(unit):
        memory_str = f'{n_bytes // K8S_UNIT_MAP[unit]}{unit}'
    return memory_str