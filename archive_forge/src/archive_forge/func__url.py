import json
import logging
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple
import requests
from ray.autoscaler._private.constants import (
from ray.autoscaler._private.util import NodeID, NodeIP, NodeKind, NodeStatus, NodeType
from ray.autoscaler.batching_node_provider import (
from ray.autoscaler.tags import (
def _url(self, path: str) -> str:
    """Convert resource path to REST URL for Kubernetes API server."""
    if path.startswith('pods'):
        api_group = '/api/v1'
    elif path.startswith('rayclusters'):
        api_group = '/apis/ray.io/' + KUBERAY_CRD_VER
    else:
        raise NotImplementedError('Tried to access unknown entity at {}'.format(path))
    return 'https://kubernetes.default:443' + api_group + '/namespaces/' + self.namespace + '/' + path