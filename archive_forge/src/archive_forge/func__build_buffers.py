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
def _build_buffers(self, single_row: Dict[str, TensorType]) -> None:
    """Builds the buffers for sample collection, given an example data row.

        Args:
            single_row (Dict[str, TensorType]): A single row (keys=column
                names) of data to base the buffers on.
        """
    for col, data in single_row.items():
        if col in self.buffers:
            continue
        shift = self.shift_before - (1 if col in [SampleBatch.OBS, SampleBatch.INFOS, SampleBatch.EPS_ID, SampleBatch.AGENT_INDEX, SampleBatch.ENV_ID, SampleBatch.T, SampleBatch.UNROLL_ID] else 0)
        should_flatten_action_key = col == SampleBatch.ACTIONS and (not self.disable_action_flattening)
        should_flatten_state_key = col.startswith('state_out') and (not self._enable_new_api_stack)
        if col == SampleBatch.INFOS or should_flatten_state_key or should_flatten_action_key:
            if should_flatten_action_key:
                data = flatten_to_single_ndarray(data)
            self.buffers[col] = [[data for _ in range(shift)]]
        else:
            self.buffers[col] = [[v for _ in range(shift)] for v in tree.flatten(data)]
            self.buffer_structs[col] = data