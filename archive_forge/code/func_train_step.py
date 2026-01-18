from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import copy
import enum
import math
import os
import signal
import sys
import threading
import time
import tensorflow as tf
import numpy as np
import six
from six.moves import queue as Queue  # pylint: disable=redefined-builtin
from six.moves import xrange  # pylint: disable=redefined-builtin
from tensorflow.core.framework import variable_pb2
from tensorflow.core.framework.summary_pb2 import Summary
from tensorflow.core.protobuf.tpu import compilation_result_pb2 as tpu_compilation_result
from tensorflow.python.data.util import nest as data_nest
from tensorflow.python.distribute.cluster_resolver import tpu_cluster_resolver
from tensorflow.python.framework import function
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import control_flow_util
from tensorflow.python.ops import ref_variable
from tensorflow.python.ops import summary_ops_v2
from tensorflow.python.ops import variable_scope
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.tpu import functional as tpu_functional
from tensorflow.python.tpu import preempted_hook
from tensorflow.python.tpu import session_support
from tensorflow.python.tpu import tensor_tracer
from tensorflow.python.tpu import tpu
from tensorflow.python.tpu import tpu_embedding_gradient
from tensorflow.python.tpu import tpu_feed
from tensorflow.python.tpu import tpu_function
from tensorflow.python.tpu import tpu_replication
from tensorflow.python.tpu import training_loop
from tensorflow.python.tpu.ops import tpu_ops
from tensorflow.python.training import evaluation
from tensorflow.python.util import function_utils
from tensorflow.python.util import tf_inspect
from tensorflow_estimator.python.estimator import estimator as estimator_lib
from tensorflow_estimator.python.estimator import model_fn as model_fn_lib
from tensorflow_estimator.python.estimator.estimator_export import estimator_export
from tensorflow_estimator.python.estimator.export import export_output as export_output_lib
from tensorflow_estimator.python.estimator.tpu import _tpu_estimator_embedding
from tensorflow_estimator.python.estimator.tpu import error_handling
from tensorflow_estimator.python.estimator.tpu import iteration_count_estimator
from tensorflow_estimator.python.estimator.tpu import tpu_config
from tensorflow_estimator.python.estimator.tpu import tpu_context
from tensorflow_estimator.python.estimator.tpu import util as util_lib
from tensorflow_estimator.python.estimator.tpu._tpu_estimator_embedding import AdagradParameters  # pylint: disable=unused-import
from tensorflow_estimator.python.estimator.tpu._tpu_estimator_embedding import AdamParameters  # pylint: disable=unused-import
from tensorflow_estimator.python.estimator.tpu._tpu_estimator_embedding import EmbeddingConfigSpec  # pylint: disable=unused-import
from tensorflow_estimator.python.estimator.tpu._tpu_estimator_embedding import StochasticGradientDescentParameters  # pylint: disable=unused-import
def train_step(step):
    """Training step function for use inside a while loop."""
    inputs = dequeue_fn()
    features, labels = inputs.features_and_labels()
    self._add_embedding_features(features, True)
    estimator_spec = self._verify_estimator_spec(self._call_model_fn(features, labels))
    loss, train_op = (estimator_spec.loss, estimator_spec.train_op)
    if tensor_tracer.TensorTracer.is_enabled():
        tt = tensor_tracer.TensorTracer()
        loss = tt.trace_tpu(tf.compat.v1.get_default_graph(), loss, train_op, self._ctx.num_replicas)
        tracer_host_call = tt.host_call_deps_and_fn()
    else:
        tracer_host_call = {}
    if isinstance(estimator_spec, model_fn_lib._TPUEstimatorSpec):
        captured_scaffold_fn.capture(estimator_spec.scaffold_fn)
    else:
        captured_scaffold_fn.capture(None)
    captured_training_hooks.capture(estimator_spec.training_hooks)
    if self._ctx.embedding_config is None:
        apply_sparse_grads = []
    else:
        tpu_embedding_ = self._ctx.embedding_config.tpu_embedding
        gradients = tpu_embedding_gradient.get_gradients_through_dummy_table_variables(tpu_embedding_)
        grad_multiplier = self._ctx.embedding_config.get_grad_multiplier()
        if grad_multiplier is not None:
            scaled_gradients = collections.OrderedDict(((k, v * grad_multiplier) for k, v in six.iteritems(gradients)))
        else:
            scaled_gradients = gradients
        apply_sparse_grads = [tpu_embedding_.generate_send_gradients_op(scaled_gradients, tf.compat.v1.train.get_global_step())]
    stopping_signals = None
    user_provided_stopping_signals_name = None
    if self._ctx.feed_hook is not None:
        stopping_signals, user_provided_stopping_signals_name = self._ctx.feed_hook.get_stopping_signals_and_name(features)
    with tf.control_dependencies([train_op] + apply_sparse_grads):
        host_call_outfeed_ops = []
        host_call_fn, host_call_args = (None, [])
        if isinstance(estimator_spec, model_fn_lib._TPUEstimatorSpec) and estimator_spec.host_call is not None:
            host_call_fn, host_call_args = estimator_spec.host_call
        if stopping_signals is not None:
            identity_fn = lambda **kwargs: kwargs
            tracer_host_call[user_provided_stopping_signals_name] = [identity_fn, stopping_signals]
        if host_call_fn:
            if host_call_args:
                tracer_host_call.update({'host_call': estimator_spec.host_call})
                host_call.record(tracer_host_call)
                host_call_outfeed_ops = host_call.create_enqueue_op(step)
            elif tracer_host_call:
                host_call.record(tracer_host_call)
                host_call_outfeed_ops = host_call.create_enqueue_op(step)
        else:
            tracer_host_call.update({'host_call': (lambda loss_t: loss_t, [tf.reshape(loss, [1])])})
            host_call.record(tracer_host_call)
            host_call_outfeed_ops = host_call.create_enqueue_op(step)
        with tf.control_dependencies(host_call_outfeed_ops):
            return tf.identity(loss)