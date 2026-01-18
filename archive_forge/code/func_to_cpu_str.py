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
def to_cpu_str(n_cpus: Union[int, float]) -> str:
    """Convert number of cpus to cpu string
    (e.g. 0.5 -> '500m')
    """
    if n_cpus == 0:
        return '0'
    millicores = n_cpus * 1000
    if millicores % 1 == 0:
        return f'{int(millicores)}m'
    microcores = n_cpus * 1000000
    if microcores % 1 == 0:
        return f'{int(microcores)}u'
    nanocores = n_cpus * 1000000000
    return f'{int(nanocores)}n'