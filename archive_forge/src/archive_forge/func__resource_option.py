import warnings
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple, Union
import ray
from ray._private import ray_constants
from ray._private.utils import get_ray_doc_version
from ray.util.placement_group import PlacementGroup
from ray.util.scheduling_strategies import (
def _resource_option(name: str, default_value: Any=None):
    """This is used for resource related options."""
    return Option((float, int, type(None)), lambda x: None if x is None else _validate_resource_quantity(name, x), default_value=default_value)