import copy
import logging
import math
from typing import Any, Dict, List, Optional
import numpy as np
import tree  # pip install dm_tree
from gymnasium.spaces import Space
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.view_requirement import ViewRequirement
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.utils.spaces.space_utils import (
from ray.rllib.utils.typing import (
from ray.util.annotations import PublicAPI
def _unflatten_as_buffer_struct(self, data: List[np.ndarray], key: str) -> np.ndarray:
    """Unflattens the given to match the buffer struct format for that key."""
    if key not in self.buffer_structs:
        return data[0]
    return tree.unflatten_as(self.buffer_structs[key], data)