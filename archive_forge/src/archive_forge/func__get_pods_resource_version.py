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
def _get_pods_resource_version(self) -> str:
    """
        Extract a recent pods resource version by reading the head pod's
        metadata.resourceVersion of the response.
        """
    if not RAY_HEAD_POD_NAME:
        return None
    pod_resp = self._get(f'pods/{RAY_HEAD_POD_NAME}')
    return pod_resp['metadata']['resourceVersion']