import abc
from copy import deepcopy
import numpy as np
from typing import Any, Optional, Dict, List, Tuple, Union, Type
from ray.rllib.utils import try_import_jax, try_import_tf, try_import_torch
from ray.rllib.utils.annotations import OverrideToImplementCustomLogic
from ray.rllib.utils.annotations import DeveloperAPI, override
from ray.rllib.utils.typing import TensorType
def _get_full_shape(self) -> Tuple[int]:
    """Converts the expected shape to a shape by replacing the unknown dimension
        sizes with a value of 1."""
    sampled_shape = tuple()
    for d in self._expected_shape:
        if isinstance(d, int):
            sampled_shape += (d,)
        else:
            sampled_shape += (1,)
    return sampled_shape