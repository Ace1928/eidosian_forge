import decimal
import json
import logging
import time
from typing import Any, Dict, Optional
import requests
from ray.autoscaler._private.constants import (
from ray.autoscaler._private.kuberay import node_provider, utils
from ray.autoscaler._private.util import validate_config
def _generate_provider_config(ray_cluster_namespace: str) -> Dict[str, Any]:
    """Generates the `provider` field of the autoscaling config, which carries data
    required to instantiate the KubeRay node provider.
    """
    return {'type': 'kuberay', 'namespace': ray_cluster_namespace, DISABLE_NODE_UPDATERS_KEY: True, DISABLE_LAUNCH_CONFIG_CHECK_KEY: True, FOREGROUND_NODE_LAUNCH_KEY: True, WORKER_LIVENESS_CHECK_KEY: False, WORKER_RPC_DRAIN_KEY: True}