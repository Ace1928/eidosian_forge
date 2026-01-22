import functools
import logging
import os
import threading
from typing import Dict, List, Optional, Tuple, Union
import tree  # pip install dm_tree
from ray.rllib.evaluation.episode import Episode
from ray.rllib.models.catalog import ModelCatalog
from ray.rllib.models.repeated_values import RepeatedValues
from ray.rllib.policy.policy import Policy, PolicyState
from ray.rllib.policy.rnn_sequencing import pad_batch_to_sequences_of_same_size
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils import add_mixins, force_list
from ray.rllib.utils.annotations import override, DeveloperAPI
from ray.rllib.utils.deprecation import (
from ray.rllib.utils.error import ERR_MSG_TF_POLICY_CANNOT_SAVE_KERAS_MODEL
from ray.rllib.utils.framework import try_import_tf
from ray.rllib.utils.metrics import (
from ray.rllib.utils.metrics.learner_info import LEARNER_STATS_KEY
from ray.rllib.utils.numpy import convert_to_numpy
from ray.rllib.utils.spaces.space_utils import normalize_action
from ray.rllib.utils.tf_utils import get_gpu_devices
from ray.rllib.utils.threading import with_lock
from ray.rllib.utils.typing import (
from ray.util.debug import log_once
@with_lock
def _compute_gradients_helper(self, samples):
    """Computes and returns grads as eager tensors."""
    self._re_trace_counter += 1
    if isinstance(self.model, tf.keras.Model):
        variables = self.model.trainable_variables
    else:
        variables = self.model.trainable_variables()
    with tf.GradientTape(persistent=compute_gradients_fn is not None) as tape:
        losses = self._loss(self, self.model, self.dist_class, samples)
    losses = force_list(losses)
    if compute_gradients_fn:
        optimizer = _OptimizerWrapper(tape)
        if self.config['_tf_policy_handles_more_than_one_loss']:
            grads_and_vars = compute_gradients_fn(self, [optimizer] * len(losses), losses)
        else:
            grads_and_vars = [compute_gradients_fn(self, optimizer, losses[0])]
    else:
        grads_and_vars = [list(zip(tape.gradient(loss, variables), variables)) for loss in losses]
    if log_once('grad_vars'):
        for g_and_v in grads_and_vars:
            for g, v in g_and_v:
                if g is not None:
                    logger.info(f'Optimizing variable {v.name}')
    if self.config['_tf_policy_handles_more_than_one_loss']:
        grads = [[g for g, _ in g_and_v] for g_and_v in grads_and_vars]
    else:
        grads_and_vars = grads_and_vars[0]
        grads = [g for g, _ in grads_and_vars]
    stats = self._stats(self, samples, grads)
    return (grads_and_vars, grads, stats)