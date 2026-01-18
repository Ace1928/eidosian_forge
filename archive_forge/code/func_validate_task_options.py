import warnings
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple, Union
import ray
from ray._private import ray_constants
from ray._private.utils import get_ray_doc_version
from ray.util.placement_group import PlacementGroup
from ray.util.scheduling_strategies import (
def validate_task_options(options: Dict[str, Any], in_options: bool):
    """Options check for Ray tasks.

    Args:
        options: Options for Ray tasks.
        in_options: If True, we are checking the options under the context of
            ".options()".
    """
    for k, v in options.items():
        if k not in task_options:
            raise ValueError(f'Invalid option keyword {k} for remote functions. Valid ones are {list(task_options)}.')
        task_options[k].validate(k, v)
    if in_options and 'max_calls' in options:
        raise ValueError("Setting 'max_calls' is not supported in '.options()'.")
    _check_deprecate_placement_group(options)