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
def _patch(self, path: str, payload: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Wrapper for REST PATCH of resource with proper headers."""
    url = url_from_resource(namespace=self.namespace, path=path)
    result = requests.patch(url, json.dumps(payload), headers={**self.headers, 'Content-type': 'application/json-patch+json'}, verify=self.verify)
    if not result.status_code == 200:
        result.raise_for_status()
    return result.json()