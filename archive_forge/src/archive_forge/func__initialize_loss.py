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
def _initialize_loss(self, losses: List[TensorType], loss_inputs: List[Tuple[str, TensorType]]) -> None:
    """Initializes the loss op from given loss tensor and placeholders.

        Args:
            loss (List[TensorType]): The list of loss ops returned by some
                loss function.
            loss_inputs (List[Tuple[str, TensorType]]): The list of Tuples:
                (name, tf1.placeholders) needed for calculating the loss.
        """
    self._loss_input_dict = dict(loss_inputs)
    self._loss_input_dict_no_rnn = {k: v for k, v in self._loss_input_dict.items() if v not in self._state_inputs and v != self._seq_lens}
    for i, ph in enumerate(self._state_inputs):
        self._loss_input_dict['state_in_{}'.format(i)] = ph
    if self.model and (not isinstance(self.model, tf.keras.Model)):
        self._losses = force_list(self.model.custom_loss(losses, self._loss_input_dict))
        self._stats_fetches.update({'model': self.model.metrics()})
    else:
        self._losses = losses
    self._loss = self._losses[0] if self._losses is not None else None
    if not self._optimizers:
        self._optimizers = force_list(self.optimizer())
        self._optimizer = self._optimizers[0] if self._optimizers else None
    if self.config['_tf_policy_handles_more_than_one_loss']:
        self._grads_and_vars = []
        self._grads = []
        for group in self.gradients(self._optimizers, self._losses):
            g_and_v = [(g, v) for g, v in group if g is not None]
            self._grads_and_vars.append(g_and_v)
            self._grads.append([g for g, _ in g_and_v])
    else:
        self._grads_and_vars = [(g, v) for g, v in self.gradients(self._optimizer, self._loss) if g is not None]
        self._grads = [g for g, _ in self._grads_and_vars]
    if self.model:
        self._variables = ray.experimental.tf_utils.TensorFlowVariables([], self.get_session(), self.variables())
    if len(self.devices) <= 1:
        if not self._update_ops:
            self._update_ops = tf1.get_collection(tf1.GraphKeys.UPDATE_OPS, scope=tf1.get_variable_scope().name)
        if self._update_ops:
            logger.info('Update ops to run on apply gradient: {}'.format(self._update_ops))
        with tf1.control_dependencies(self._update_ops):
            self._apply_op = self.build_apply_op(optimizer=self._optimizers if self.config['_tf_policy_handles_more_than_one_loss'] else self._optimizer, grads_and_vars=self._grads_and_vars)
    if log_once('loss_used'):
        logger.debug(f'These tensors were used in the loss functions:\n{summarize(self._loss_input_dict)}\n')
    self.get_session().run(tf1.global_variables_initializer())
    self._optimizer_variables = ray.experimental.tf_utils.TensorFlowVariables([v for o in self._optimizers for v in o.variables()], self.get_session())