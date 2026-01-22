import warnings
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple, Union
import ray
from ray._private import ray_constants
from ray._private.utils import get_ray_doc_version
from ray.util.placement_group import PlacementGroup
from ray.util.scheduling_strategies import (
@dataclass
class Option:
    type_constraint: Optional[Union[type, Tuple[type]]] = None
    value_constraint: Optional[Callable[[Any], Optional[str]]] = None
    default_value: Any = None

    def validate(self, keyword: str, value: Any):
        """Validate the option."""
        if self.type_constraint is not None:
            if not isinstance(value, self.type_constraint):
                raise TypeError(f"The type of keyword '{keyword}' must be {self.type_constraint}, but received type {type(value)}")
        if self.value_constraint is not None:
            possible_error_message = self.value_constraint(value)
            if possible_error_message:
                raise ValueError(possible_error_message)