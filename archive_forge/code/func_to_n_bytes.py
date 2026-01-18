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
def to_n_bytes(memory_str: str) -> int:
    """Convert memory string to number of bytes
    (e.g. '1234Mi'-> 1293942784)
    """
    if memory_str.startswith('0'):
        return 0
    if memory_str.isnumeric():
        return int(memory_str)
    for unit, multiplier in K8S_UNIT_MAP.items():
        if memory_str.endswith(unit):
            return int(memory_str.strip(unit)) * multiplier