import logging
import math
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union
import gymnasium as gym
import numpy as np
import tree  # pip install dm_tree
import ray
import ray.experimental.tf_utils
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.policy.policy import Policy, PolicyState, PolicySpec
from ray.rllib.policy.rnn_sequencing import pad_batch_to_sequences_of_same_size
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils import force_list
from ray.rllib.utils.annotations import DeveloperAPI, override
from ray.rllib.utils.debug import summarize
from ray.rllib.utils.deprecation import Deprecated
from ray.rllib.utils.error import ERR_MSG_TF_POLICY_CANNOT_SAVE_KERAS_MODEL
from ray.rllib.utils.framework import try_import_tf
from ray.rllib.utils.metrics import (
from ray.rllib.utils.metrics.learner_info import LEARNER_STATS_KEY
from ray.rllib.utils.spaces.space_utils import normalize_action
from ray.rllib.utils.tf_run_builder import _TFRunBuilder
from ray.rllib.utils.tf_utils import get_gpu_devices
from ray.rllib.utils.typing import (
from ray.util.debug import log_once
def _get_loss_inputs_dict(self, train_batch: SampleBatch, shuffle: bool):
    """Return a feed dict from a batch.

        Args:
            train_batch: batch of data to derive inputs from.
            shuffle: whether to shuffle batch sequences. Shuffle may
                be done in-place. This only makes sense if you're further
                applying minibatch SGD after getting the outputs.

        Returns:
            Feed dict of data.
        """
    if not isinstance(train_batch, SampleBatch) or not train_batch.zero_padded:
        pad_batch_to_sequences_of_same_size(train_batch, max_seq_len=self._max_seq_len, shuffle=shuffle, batch_divisibility_req=self._batch_divisibility_req, feature_keys=list(self._loss_input_dict_no_rnn.keys()), view_requirements=self.view_requirements)
    train_batch.set_training(True)
    feed_dict = {}
    for key, placeholders in self._loss_input_dict.items():
        a = tree.map_structure(lambda ph, v: feed_dict.__setitem__(ph, v), placeholders, train_batch[key])
        del a
    state_keys = ['state_in_{}'.format(i) for i in range(len(self._state_inputs))]
    for key in state_keys:
        feed_dict[self._loss_input_dict[key]] = train_batch[key]
    if state_keys:
        feed_dict[self._seq_lens] = train_batch[SampleBatch.SEQ_LENS]
    return feed_dict