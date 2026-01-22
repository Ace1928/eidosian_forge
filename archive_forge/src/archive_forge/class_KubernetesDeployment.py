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
class KubernetesDeployment:

    def __init__(self, id: str, name: str, namespace: str, created_at: str, replicas: int, selector: Dict[str, Any], extra: Optional[Dict[str, Any]]=None):
        self.id = id
        self.name = name
        self.namespace = namespace
        self.created_at = created_at
        self.replicas = replicas
        self.selector = selector
        self.extra = extra or {}

    def __repr__(self):
        return '<KubernetesDeployment name={} namespace={} replicas={}>'.format(self.name, self.namespace, self.replicas)