import decimal
import json
import logging
import time
from typing import Any, Dict, Optional
import requests
from ray.autoscaler._private.constants import (
from ray.autoscaler._private.kuberay import node_provider, utils
from ray.autoscaler._private.util import validate_config
def _get_memory(ray_start_params: Dict[str, str], k8s_resource_limits: Dict[str, Any]) -> Optional[int]:
    """Get memory resource annotation from ray_start_params or k8s_resource_limits,
    with priority for ray_start_params.
    """
    if 'memory' in ray_start_params:
        return int(ray_start_params['memory'])
    elif 'memory' in k8s_resource_limits:
        memory_quantity: str = k8s_resource_limits['memory']
        return _round_up_k8s_quantity(memory_quantity)
    return None