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
def _actual_train_model_distributed(self, strategy, input_fn, hooks, saving_listeners):
    """That method that does actual training with distribution strategy."""
    is_tpu_strategy = strategy.__class__.__name__.startswith('TPUStrategy')
    worker_hooks = []
    with tf.Graph().as_default() as g:
        if is_tpu_strategy:
            steps_per_run_variable = training.get_or_create_steps_per_run_variable()
        if hasattr(strategy, '_scale_loss_for_estimator_enabled'):
            scale_ctx = strategy._scale_loss_for_estimator_enabled()
        else:

            @tf_contextlib.contextmanager
            def nullcontextmanager():
                yield
            scale_ctx = nullcontextmanager()
        with strategy.scope(), scale_ctx:
            tf.compat.v1.random.set_random_seed(self._config.tf_random_seed)
            iterator, input_hooks = self._get_iterator_from_input_fn(input_fn, ModeKeys.TRAIN, strategy)
            worker_hooks.extend(input_hooks)
            global_step_tensor = self._create_and_assert_global_step(g)
            tf.compat.v1.add_to_collection(training_util.GLOBAL_STEP_READ_KEY, strategy.extended.read_var(global_step_tensor))
            if is_tpu_strategy:

                def step_fn(ctx, inputs):
                    """A single step that is passed to run_on_dataset."""
                    if isinstance(inputs, tuple):
                        features, labels = inputs
                    else:
                        features = inputs
                        labels = None
                    estimator_spec = strategy.extended.call_for_each_replica(self._call_model_fn, args=(features, labels, ModeKeys.TRAIN, self.config))
                    ctx.set_last_step_output(name='loss', output=estimator_spec.loss, reduce_op=_get_loss_reduce_op_for_reporting())
                    ctx.set_non_tensor_output(name='estimator_spec', output=estimator_spec)
                    return estimator_spec.train_op
                initial_training_loss = tf.constant(10000000.0)
                ctx = strategy.extended.experimental_run_steps_on_iterator(step_fn, iterator, iterations=steps_per_run_variable, initial_loop_values={'loss': initial_training_loss})
                distributed_train_op = ctx.run_op
                loss = ctx.last_step_outputs['loss']
                grouped_estimator_spec = ctx.non_tensor_outputs['estimator_spec']
            else:
                features, labels = estimator_util.parse_iterator_result(iterator.get_next())
                grouped_estimator_spec = strategy.extended.call_for_each_replica(self._call_model_fn, args=(features, labels, ModeKeys.TRAIN, self.config))
                loss = strategy.reduce(_get_loss_reduce_op_for_reporting(), grouped_estimator_spec.loss, axis=None)
                distributed_train_op = grouped_estimator_spec.train_op
            scaffold = _combine_distributed_scaffold(grouped_estimator_spec.scaffold, strategy)

            def get_hooks_from_the_first_device(per_device_hooks):
                return [self._train_distribution.experimental_local_results(per_device_hook)[0] for per_device_hook in per_device_hooks]
            training_hooks = get_hooks_from_the_first_device(grouped_estimator_spec.training_hooks)
            training_chief_hooks = get_hooks_from_the_first_device(grouped_estimator_spec.training_chief_hooks)
            estimator_spec = model_fn_lib.EstimatorSpec(mode=grouped_estimator_spec.mode, loss=loss, train_op=strategy.group(distributed_train_op), training_hooks=training_hooks, training_chief_hooks=training_chief_hooks, scaffold=scaffold)
            return self._train_with_estimator_spec(estimator_spec, worker_hooks, hooks, global_step_tensor, saving_listeners)