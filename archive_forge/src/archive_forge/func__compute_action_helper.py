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
@with_lock
def _compute_action_helper(self, input_dict, state_batches, seq_lens, explore, timestep):
    """Shared forward pass logic (w/ and w/o trajectory view API).

        Returns:
            A tuple consisting of a) actions, b) state_out, c) extra_fetches.
        """
    explore = explore if explore is not None else self.config['explore']
    timestep = timestep if timestep is not None else self.global_timestep
    self._is_recurrent = state_batches is not None and state_batches != []
    if self.model:
        self.model.eval()
    if self.action_sampler_fn:
        action_dist = dist_inputs = None
        action_sampler_outputs = self.action_sampler_fn(self, self.model, input_dict, state_batches, explore=explore, timestep=timestep)
        if len(action_sampler_outputs) == 4:
            actions, logp, dist_inputs, state_out = action_sampler_outputs
        else:
            actions, logp, state_out = action_sampler_outputs
    else:
        self.exploration.before_compute_actions(explore=explore, timestep=timestep)
        if self.action_distribution_fn:
            try:
                dist_inputs, dist_class, state_out = self.action_distribution_fn(self, self.model, input_dict=input_dict, state_batches=state_batches, seq_lens=seq_lens, explore=explore, timestep=timestep, is_training=False)
            except TypeError as e:
                if 'positional argument' in e.args[0] or 'unexpected keyword argument' in e.args[0]:
                    dist_inputs, dist_class, state_out = self.action_distribution_fn(self, self.model, input_dict[SampleBatch.CUR_OBS], explore=explore, timestep=timestep, is_training=False)
                else:
                    raise e
        else:
            dist_class = self.dist_class
            dist_inputs, state_out = self.model(input_dict, state_batches, seq_lens)
        if not (isinstance(dist_class, functools.partial) or issubclass(dist_class, TorchDistributionWrapper)):
            raise ValueError('`dist_class` ({}) not a TorchDistributionWrapper subclass! Make sure your `action_distribution_fn` or `make_model_and_action_dist` return a correct distribution class.'.format(dist_class.__name__))
        action_dist = dist_class(dist_inputs, self.model)
        actions, logp = self.exploration.get_exploration_action(action_distribution=action_dist, timestep=timestep, explore=explore)
    input_dict[SampleBatch.ACTIONS] = actions
    extra_fetches = self.extra_action_out(input_dict, state_batches, self.model, action_dist)
    if dist_inputs is not None:
        extra_fetches[SampleBatch.ACTION_DIST_INPUTS] = dist_inputs
    if logp is not None:
        extra_fetches[SampleBatch.ACTION_PROB] = torch.exp(logp.float())
        extra_fetches[SampleBatch.ACTION_LOGP] = logp
    self.global_timestep += len(input_dict[SampleBatch.CUR_OBS])
    return convert_to_numpy((actions, state_out, extra_fetches))