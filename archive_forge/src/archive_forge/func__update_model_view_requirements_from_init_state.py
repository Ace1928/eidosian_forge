import json
import logging
import os
import platform
from abc import ABCMeta, abstractmethod
from typing import (
import gymnasium as gym
import numpy as np
import tree  # pip install dm_tree
from gymnasium.spaces import Box
from packaging import version
import ray
import ray.cloudpickle as pickle
from ray.actor import ActorHandle
from ray.train import Checkpoint
from ray.rllib.core.models.base import STATE_IN, STATE_OUT
from ray.rllib.models.action_dist import ActionDistribution
from ray.rllib.models.catalog import ModelCatalog
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.policy.rnn_sequencing import add_time_dimension
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.view_requirement import ViewRequirement
from ray.rllib.utils.annotations import (
from ray.rllib.utils.checkpoints import (
from ray.rllib.utils.deprecation import (
from ray.rllib.utils.exploration.exploration import Exploration
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.utils.from_config import from_config
from ray.rllib.utils.numpy import convert_to_numpy
from ray.rllib.utils.serialization import (
from ray.rllib.utils.spaces.space_utils import (
from ray.rllib.utils.tensor_dtype import get_np_dtype
from ray.rllib.utils.tf_utils import get_tf_eager_cls_if_necessary
from ray.rllib.utils.typing import (
from ray.util.annotations import PublicAPI
def _update_model_view_requirements_from_init_state(self):
    """Uses Model's (or this Policy's) init state to add needed ViewReqs.

        Can be called from within a Policy to make sure RNNs automatically
        update their internal state-related view requirements.
        Changes the `self.view_requirements` dict.
        """
    self._model_init_state_automatically_added = True
    model = getattr(self, 'model', None)
    obj = model or self
    if model and (not hasattr(model, 'view_requirements')):
        model.view_requirements = {SampleBatch.OBS: ViewRequirement(space=self.observation_space)}
    view_reqs = obj.view_requirements
    init_state = []
    if hasattr(obj, 'get_initial_state') and callable(obj.get_initial_state):
        init_state = obj.get_initial_state()
    elif tf and isinstance(model, tf.keras.Model) and ('state_in_0' not in view_reqs):
        obj.get_initial_state = lambda: [np.zeros_like(view_req.space.sample()) for k, view_req in model.view_requirements.items() if k.startswith('state_in_')]
    else:
        obj.get_initial_state = lambda: []
        if 'state_in_0' in view_reqs:
            self.is_recurrent = lambda: True
    view_reqs = [view_reqs] + ([self.view_requirements] if hasattr(self, 'view_requirements') else [])
    for i, state in enumerate(init_state):
        fw = np if isinstance(state, np.ndarray) else torch if torch and torch.is_tensor(state) else None
        if fw:
            space = Box(-1.0, 1.0, shape=state.shape) if fw.all(state == 0.0) else state
        else:
            space = state
        for vr in view_reqs:
            if 'state_in_{}'.format(i) not in vr:
                vr['state_in_{}'.format(i)] = ViewRequirement('state_out_{}'.format(i), shift=-1, used_for_compute_actions=True, batch_repeat_value=self.config.get('model', {}).get('max_seq_len', 1), space=space)
            if 'state_out_{}'.format(i) not in vr:
                vr['state_out_{}'.format(i)] = ViewRequirement(space=space, used_for_training=True)