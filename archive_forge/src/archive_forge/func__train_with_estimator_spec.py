from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import copy
import os
import tempfile
import numpy as np
import six
import tensorflow as tf
from google.protobuf import message
from tensorflow.core.framework import summary_pb2
from tensorflow.python.checkpoint import checkpoint as trackable_util
from tensorflow.python.checkpoint import checkpoint_management
from tensorflow.python.checkpoint import graph_view
from tensorflow.python.distribute import estimator_training as distribute_coordinator_training
from tensorflow.python.eager import context
from tensorflow.python.eager import monitoring
from tensorflow.python.framework import ops
from tensorflow.python.profiler import trace
from tensorflow.python.saved_model import path_helpers
from tensorflow.python.summary import summary
from tensorflow.python.training import basic_session_run_hooks
from tensorflow.python.training import device_setter
from tensorflow.python.training import evaluation
from tensorflow.python.training import training
from tensorflow.python.training import training_util
from tensorflow.python.util import deprecation
from tensorflow.python.util import function_utils
from tensorflow.python.util import tf_contextlib
from tensorflow.tools.docs import doc_controls
from tensorflow_estimator.python.estimator import model_fn as model_fn_lib
from tensorflow_estimator.python.estimator import run_config
from tensorflow_estimator.python.estimator import util as estimator_util
from tensorflow_estimator.python.estimator.estimator_export import estimator_export
from tensorflow_estimator.python.estimator.export import export_lib
from tensorflow_estimator.python.estimator.mode_keys import ModeKeys
def _train_with_estimator_spec(self, estimator_spec, worker_hooks, hooks, global_step_tensor, saving_listeners):
    """Train a model with the given Estimator Spec."""
    if self._warm_start_settings and (not tf.train.latest_checkpoint(self._model_dir)):
        tf.compat.v1.logging.info('Warm-starting with WarmStartSettings: %s' % (self._warm_start_settings,))
        tf.compat.v1.train.warm_start(*self._warm_start_settings)
    if not any([x.op.name == 'loss' for x in ops.get_collection(ops.GraphKeys.SUMMARIES)]):
        summary.scalar('loss', estimator_spec.loss)
    ops.add_to_collection(ops.GraphKeys.LOSSES, estimator_spec.loss)
    worker_hooks.extend(hooks)
    worker_hooks.append(tf.compat.v1.train.NanTensorHook(estimator_spec.loss))
    if self._config.log_step_count_steps is not None:
        worker_hooks.append(tf.compat.v1.train.LoggingTensorHook({'loss': estimator_spec.loss, 'step': global_step_tensor}, every_n_iter=self._config.log_step_count_steps))
    worker_hooks.extend(estimator_spec.training_hooks)
    if not (estimator_spec.scaffold.saver or tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.SAVERS)):
        tf.compat.v1.add_to_collection(tf.compat.v1.GraphKeys.SAVERS, tf.compat.v1.train.Saver(sharded=True, max_to_keep=self._config.keep_checkpoint_max, keep_checkpoint_every_n_hours=self._config.keep_checkpoint_every_n_hours, defer_build=True, save_relative_paths=True))
    if self._config.cluster_spec and type(self._train_distribution).__name__ in ('CollectiveAllReduceStrategy', 'CollectiveAllReduceStrategyV1', 'MultiWorkerMirroredStrategy'):
        return self._train_with_estimator_spec_distributed(estimator_spec, worker_hooks, saving_listeners)
    chief_hooks = []
    all_hooks = worker_hooks + list(estimator_spec.training_chief_hooks)
    saver_hooks = [h for h in all_hooks if isinstance(h, tf.compat.v1.train.CheckpointSaverHook)]
    if self._config.save_checkpoints_secs or self._config.save_checkpoints_steps:
        if not saver_hooks:
            chief_hooks = [tf.compat.v1.train.CheckpointSaverHook(self._model_dir, save_secs=self._config.save_checkpoints_secs, save_steps=self._config.save_checkpoints_steps, scaffold=estimator_spec.scaffold, save_graph_def=self._config.checkpoint_save_graph_def)]
            saver_hooks = [chief_hooks[0]]
    if saving_listeners:
        if not saver_hooks:
            raise ValueError('There should be a CheckpointSaverHook to use saving_listeners. Please set one of the RunConfig.save_checkpoints_steps or RunConfig.save_checkpoints_secs.')
        else:
            for listener in saving_listeners:
                if listener not in saver_hooks[0]._listeners:
                    saver_hooks[0]._listeners.append(listener)
    save_summary_steps = self._config.save_summary_steps
    log_step_count_steps = self._config.log_step_count_steps
    if self._config.cluster_spec and self._config.cluster_spec.jobs and (run_config.TaskType.WORKER in self._config.cluster_spec.jobs) and (run_config.TaskType.MASTER in self._config.cluster_spec.jobs):
        save_summary_steps = 0
        log_step_count_steps = None
        if self._config.task_type == run_config.TaskType.WORKER and self._config.task_id == 0:
            if self._config.save_summary_steps and self._config.save_summary_steps > 0:
                worker_hooks.append(tf.compat.v1.train.SummarySaverHook(save_steps=self._config.save_summary_steps, output_dir=self._config.model_dir, scaffold=estimator_spec.scaffold))
            if self._config.log_step_count_steps and self._config.log_step_count_steps > 0:
                worker_hooks.append(tf.compat.v1.train.StepCounterHook(every_n_steps=self._config.log_step_count_steps, output_dir=self._config.model_dir))
    with training.MonitoredTrainingSession(master=self._config.master, is_chief=self._config.is_chief, checkpoint_dir=self._model_dir, scaffold=estimator_spec.scaffold, hooks=worker_hooks, chief_only_hooks=tuple(chief_hooks) + tuple(estimator_spec.training_chief_hooks), save_checkpoint_secs=0, save_summaries_steps=save_summary_steps, config=self._session_config, max_wait_secs=self._config.session_creation_timeout_secs, log_step_count_steps=log_step_count_steps, save_graph_def=self._config.checkpoint_save_graph_def) as mon_sess:
        loss = None
        current_step = 0
        while not mon_sess.should_stop():
            current_step += 1
            with trace.Trace('train', step_num=current_step, _r=1):
                _, loss = mon_sess.run([estimator_spec.train_op, estimator_spec.loss])
        if current_step == 0:
            tf.compat.v1.logging.warn('Training with estimator made no steps. Perhaps input is empty or misspecified.')
    return loss