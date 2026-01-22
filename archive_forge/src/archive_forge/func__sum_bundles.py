import abc
import json
from copy import deepcopy
from inspect import signature
from typing import Dict, List, Union
from dataclasses import dataclass
import ray
from ray.util import placement_group
from ray.util.annotations import DeveloperAPI
def _sum_bundles(bundles: List[Dict[str, float]]) -> Dict[str, float]:
    """Sum all resources in a list of resource bundles.

    Args:
        bundles: List of resource bundles.

    Returns: Dict containing all resources summed up.
    """
    resources = {}
    for bundle in bundles:
        for k, v in bundle.items():
            resources[k] = resources.get(k, 0) + v
    return resources