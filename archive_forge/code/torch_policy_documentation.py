import copy
import functools
import logging
import math
import os
import threading
import time
from typing import (
import gymnasium as gym
import numpy as np
import tree  # pip install dm_tree
import ray
from ray.rllib.models.catalog import ModelCatalog
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.torch.torch_action_dist import TorchDistributionWrapper
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.policy.policy import Policy, PolicyState
from ray.rllib.policy.rnn_sequencing import pad_batch_to_sequences_of_same_size
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils import NullContextManager, force_list
from ray.rllib.utils.annotations import DeveloperAPI, override
from ray.rllib.utils.error import ERR_MSG_TORCH_POLICY_CANNOT_SAVE_MODEL
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.metrics import (
from ray.rllib.utils.metrics.learner_info import LEARNER_STATS_KEY
from ray.rllib.utils.numpy import convert_to_numpy
from ray.rllib.utils.spaces.space_utils import normalize_action
from ray.rllib.utils.threading import with_lock
from ray.rllib.utils.torch_utils import convert_to_torch_tensor
from ray.rllib.utils.typing import (
Performs a parallelized loss and gradient calculation over the batch.

        Splits up the given train batch into n shards (n=number of this
        Policy's devices) and passes each data shard (in parallel) through
        the loss function using the individual devices' models
        (self.model_gpu_towers). Then returns each tower's outputs.

        Args:
            sample_batches: A list of SampleBatch shards to
                calculate loss and gradients for.

        Returns:
            A list (one item per device) of 2-tuples, each with 1) gradient
            list and 2) grad info dict.
        