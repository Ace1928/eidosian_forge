import warnings
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple, Union
import ray
from ray._private import ray_constants
from ray._private.utils import get_ray_doc_version
from ray.util.placement_group import PlacementGroup
from ray.util.scheduling_strategies import (
Validate the option.