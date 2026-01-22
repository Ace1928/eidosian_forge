import decimal
import json
import logging
import time
from typing import Any, Dict, Optional
import requests
from ray.autoscaler._private.constants import (
from ray.autoscaler._private.kuberay import node_provider, utils
from ray.autoscaler._private.util import validate_config
def _generate_available_node_types_from_ray_cr_spec(ray_cr_spec: Dict[str, Any]) -> Dict[str, Any]:
    """Formats autoscaler "available_node_types" field based on the Ray CR's group
    specs.
    """
    headGroupSpec = ray_cr_spec['headGroupSpec']
    return {_HEAD_GROUP_NAME: _node_type_from_group_spec(headGroupSpec, is_head=True), **{worker_group_spec['groupName']: _node_type_from_group_spec(worker_group_spec, is_head=False) for worker_group_spec in ray_cr_spec['workerGroupSpecs']}}