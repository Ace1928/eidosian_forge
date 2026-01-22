import abc
from copy import deepcopy
import numpy as np
from typing import Any, Optional, Dict, List, Tuple, Union, Type
from ray.rllib.utils import try_import_jax, try_import_tf, try_import_torch
from ray.rllib.utils.annotations import OverrideToImplementCustomLogic
from ray.rllib.utils.annotations import DeveloperAPI, override
from ray.rllib.utils.typing import TensorType
def _parse_expected_shape(self, shape: str, shape_vals: Dict[str, int]) -> tuple:
    """Converts the input shape to a tuple of integers and strings."""
    d_names = shape.replace(' ', '').split(',')
    self._validate_shape_vals(d_names, shape_vals)
    expected_shape = tuple((shape_vals.get(d, d) for d in d_names))
    return expected_shape