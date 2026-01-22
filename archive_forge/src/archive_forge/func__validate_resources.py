import warnings
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple, Union
import ray
from ray._private import ray_constants
from ray._private.utils import get_ray_doc_version
from ray.util.placement_group import PlacementGroup
from ray.util.scheduling_strategies import (
def _validate_resources(resources: Optional[Dict[str, float]]) -> Optional[str]:
    if resources is None:
        return None
    if 'CPU' in resources or 'GPU' in resources:
        return "Use the 'num_cpus' and 'num_gpus' keyword instead of 'CPU' and 'GPU' in 'resources' keyword"
    for name, quantity in resources.items():
        possible_error_message = _validate_resource_quantity(name, quantity)
        if possible_error_message:
            return possible_error_message
    return None