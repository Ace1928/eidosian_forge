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
Produces two functions to fold/unfold any Tensors in a struct.

    Args:
        b_dim: The batch dimension to use for folding.
        t_dim: The time dimension to use for folding.
        framework: The framework to use for folding. One of "tf2" or "torch".

    Returns:
        fold: A function that takes a struct of torch.Tensors and reshapes
            them to have a first dimension of `b_dim * t_dim`.
        unfold: A function that takes a struct of torch.Tensors and reshapes
            them to have a first dimension of `b_dim` and a second dimension
            of `t_dim`.
    