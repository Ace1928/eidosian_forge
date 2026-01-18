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
def ts_to_str(timestamp):
    """
    Return a timestamp as a nicely formatted datetime string.
    """
    date = datetime.datetime.fromtimestamp(timestamp)
    date_string = date.strftime('%d/%m/%Y %H:%M %Z')
    return date_string