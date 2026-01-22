import warnings
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple, Union
import ray
from ray._private import ray_constants
from ray._private.utils import get_ray_doc_version
from ray.util.placement_group import PlacementGroup
from ray.util.scheduling_strategies import (
def _counting_option(name: str, infinite: bool=True, default_value: Any=None):
    """This is used for positive and discrete options.

    Args:
        name: The name of the option keyword.
        infinite: If True, user could use -1 to represent infinity.
        default_value: The default value for this option.
    """
    if infinite:
        return Option((int, type(None)), lambda x: None if x is None or x >= -1 else f"The keyword '{name}' only accepts None, 0, -1 or a positive integer, where -1 represents infinity.", default_value=default_value)
    return Option((int, type(None)), lambda x: None if x is None or x >= 0 else f"The keyword '{name}' only accepts None, 0 or a positive integer.", default_value=default_value)