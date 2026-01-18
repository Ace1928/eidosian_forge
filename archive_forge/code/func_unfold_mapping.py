import logging
import numpy as np
import tree  # pip install dm_tree
from typing import List, Optional
import functools
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.annotations import DeveloperAPI
from ray.rllib.utils.debug import summarize
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.utils.typing import TensorType, ViewRequirementsDict
from ray.util import log_once
from ray.rllib.utils.typing import SampleBatchType
def unfold_mapping(item):
    if item is None:
        return item
    item = torch.as_tensor(item)
    size = list(item.size())
    current_b_dim = size[0]
    other_dims = size[1:]
    assert current_b_dim == b_dim * t_dim, 'The first dimension of the tensor must be equal to the product of the desired batch and time dimensions. Got {} and {}.'.format(current_b_dim, b_dim * t_dim)
    return item.reshape([b_dim, t_dim] + other_dims)