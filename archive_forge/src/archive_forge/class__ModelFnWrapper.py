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
class _ModelFnWrapper(object):
    """A `model_fn` wrapper.

  This makes calling model_fn on CPU and TPU easier and more consistent and
  performs necessary check and mutation required by TPU training and evaluation.

  In addition, this wrapper manages converting the `model_fn` to a single TPU
  train and eval step.
  """

    def __init__(self, model_fn, config, params, ctx):
        self._model_fn = model_fn
        self._config = config
        self._params = params
        self._ctx = ctx

    def call_without_tpu(self, features, labels, is_export_mode):
        return self._call_model_fn(features, labels, is_export_mode=is_export_mode)

    def _add_embedding_features(self, features, hook_dummy_table_variables):
        """Add embedding features, optionally add hook to intercept gradient."""
        if self._ctx.embedding_config:
            tpu_embedding_ = self._ctx.embedding_config.tpu_embedding
            embedding_activations = tpu_embedding_.get_activations()
            if hook_dummy_table_variables:
                new_embedding_activations = tpu_embedding_gradient.hook_dummy_table_variables_to_activations(tpu_embedding_, embedding_activations, self._ctx.embedding_config.dummy_table_variables)
                features.update(new_embedding_activations)
            else:
                features.update(embedding_activations)

    def convert_to_single_tpu_train_step(self, dequeue_fn):
        """Converts user provided model_fn` as a single train step on TPU.

    The user provided `model_fn` takes input tuple
    (features, labels) and produces the EstimatorSpec with train_op and loss for
    train `mode`. This usually represents a single train computation on CPU.

    For TPU training, a train (computation) step is first wrapped in a
    tf.while_loop control flow to repeat for many times and then replicated to
    all TPU shards. Besides the input should be taken from TPU infeed rather
    than input pipeline (input_fn) directly. To fit TPU loop and replicate
    pattern, the original train computation should be reformed, which is the
    returned `train_step`.

    Args:
      dequeue_fn: The function to retrieve inputs, features and labels, from TPU
        infeed dequeue channel.

    Returns:
      A tuple of train_fn, host_calls, and captured scaffold_fn. The train_fn
      representing the train step for TPU.
    """
        host_call = _OutfeedHostCall(self._ctx, outfeed_every_n_steps=self._config.tpu_config.experimental_host_call_every_n_steps)
        captured_scaffold_fn = _CapturedObject()
        captured_training_hooks = _CapturedObject()

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
        return (train_step, host_call, captured_scaffold_fn, captured_training_hooks)

    def convert_to_single_tpu_eval_step(self, dequeue_fn):
        """Converts user provided model_fn` as a single eval step on TPU.

    Similar to training, the user provided `model_fn` takes input tuple
    (features, labels) and produces the TPUEstimatorSpec with eval_metrics for
    eval `mode`. This usually represents a single evaluation computation on CPU.

    For TPU evaluation, a eval (computation) step is first wrapped in a
    tf.while_loop control flow to repeat for many times and then replicated to
    all TPU shards. Besides the input and output are slightly different. Input,
    features and labels, should be taken from TPU infeed rather than input
    pipeline (input_fn) directly. Output is managed in two stages.  First, the
    model outputs as the result of evaluation computation, usually model logits,
    should be transferred from TPU system to CPU. Then, all model outputs are
    concatenated first on CPU and sent to the metric_fn for metrics computation.
    To fit TPU evaluation pattern, the original eval computation should be
    reformed, which is the returned `eval_step`.

    Args:
      dequeue_fn: The function to retrieve inputs, features and labels, from TPU
        infeed dequeue channel.

    Returns:
      A tuple of eval_fn, host_calls, and captured scaffold_fn. The eval_fn
      representing the eval step for TPU.
    """
        host_calls = _OutfeedHostCall(self._ctx)
        captured_scaffold_fn = _CapturedObject()
        captured_eval_hooks = _CapturedObject()

        def eval_step(total_loss):
            """Evaluation step function for use inside a while loop."""
            inputs = dequeue_fn()
            features, labels = inputs.features_and_labels()
            self._add_embedding_features(features, False)
            tpu_estimator_spec = self._call_model_fn(features, labels)
            if not isinstance(tpu_estimator_spec, model_fn_lib._TPUEstimatorSpec):
                raise RuntimeError('estimator_spec used by TPU evaluation must have type`TPUEstimatorSpec`. Got {}'.format(type(tpu_estimator_spec)))
            loss = tpu_estimator_spec.loss
            captured_scaffold_fn.capture(tpu_estimator_spec.scaffold_fn)
            captured_eval_hooks.capture(tpu_estimator_spec.evaluation_hooks)
            to_record = {}
            if tpu_estimator_spec.eval_metrics:
                to_record['eval_metrics'] = tpu_estimator_spec.eval_metrics
            if tpu_estimator_spec.host_call is not None:
                to_record['host_call'] = tpu_estimator_spec.host_call
            host_calls.record(to_record)
            with tf.control_dependencies(host_calls.create_enqueue_op()):
                return tf.math.add(total_loss, loss)
        return (eval_step, host_calls, captured_scaffold_fn, captured_eval_hooks)

    def convert_to_single_tpu_predict_step(self, dequeue_fn):
        """Converts user provided model_fn` as a single predict step on TPU.

    Args:
      dequeue_fn: The function to retrieve inputs, features and labels, from TPU
        infeed dequeue channel.

    Returns:
      A tuple of predict_fn, host_calls, and captured scaffold_fn. The
      predict_fn representing the predict step for TPU.
    """
        host_calls = _OutfeedHostCall(self._ctx)
        captured_scaffold_fn = _CapturedObject()
        captured_predict_hooks = _CapturedObject()

        def predict_step(unused_scalar_stopping_signal):
            """Evaluation step function for use inside a while loop."""
            inputs = dequeue_fn()
            features, labels = inputs.features_and_labels()
            stopping_signals = inputs.signals()
            assert stopping_signals is not None, 'Internal Error: `signals` is missing.'
            tpu_estimator_spec = self._call_model_fn(features, labels, is_export_mode=False)
            if not isinstance(tpu_estimator_spec, model_fn_lib._TPUEstimatorSpec):
                raise RuntimeError('estimator_spec used by TPU prediction must have type`TPUEstimatorSpec`. Got {}'.format(type(tpu_estimator_spec)))
            self._verify_tpu_spec_predictions(tpu_estimator_spec.predictions)
            captured_scaffold_fn.capture(tpu_estimator_spec.scaffold_fn)
            captured_predict_hooks.capture(tpu_estimator_spec.prediction_hooks)
            to_record = {}
            identity_fn = lambda **kwargs: kwargs
            to_record['predictions'] = [identity_fn, tpu_estimator_spec.predictions]
            to_record['signals'] = [identity_fn, stopping_signals]
            if tpu_estimator_spec.host_call is not None:
                to_record['host_call'] = tpu_estimator_spec.host_call
            host_calls.record(to_record)
            with tf.control_dependencies(host_calls.create_enqueue_op()):
                return _StopSignals.as_scalar_stopping_signal(stopping_signals)
        return (predict_step, host_calls, captured_scaffold_fn, captured_predict_hooks)

    def _verify_tpu_spec_predictions(self, predictions):
        """Validates TPUEstimatorSpec.predictions dict."""
        if not isinstance(predictions, dict):
            raise TypeError('TPUEstimatorSpec.predictions must be dict of Tensors.')
        for key, tensor in predictions.items():
            if tensor.shape.dims[0].value is None:
                raise ValueError('The tensor with key ({}) in TPUEstimatorSpec.predictions has dynamic shape (should be static). Tensor: {}'.format(key, tensor))
        return predictions

    def _validate_model_features_and_labels(self, features, labels, is_export_mode):
        """Validates that the features and labels for the model function are valid.

    A valid features/labels object is the one with:
    - Type: A tensor or any nested structure of tensors supported by TF nest,
        namely nested dictionary, tuple, namedtuple, or sequence of tensors.
    - Static shape if is_export_mode is False.

    Args:
      features: the features that would be input to the model function.
      labels: the labels that would be input to the model function.
      is_export_mode: boolean value specifying if in export mode.

    Raises:
      TypeError: If features/labels are not of the correct type.
      ValueError: If features/labels have dynamic shape.
    """

        def validate(obj, obj_name):
            """Helper validate function."""
            if is_export_mode or self._ctx.is_running_on_cpu(is_export_mode):
                return
            if isinstance(obj, tf.Tensor):
                if not obj.get_shape().is_fully_defined():
                    raise ValueError('The {} to the model returned by input_fn must have static shape. Tensor: {}'.format(obj_name, obj))
            else:
                for tensor in data_nest.flatten(obj):
                    if not tensor.get_shape().is_fully_defined():
                        raise ValueError('The {} to the model returned by input_fn must have static shape. Tensor: {}'.format(obj_name, tensor))
        validate(features, 'features')
        if labels is not None:
            validate(labels, 'labels')

    def _call_model_fn(self, features, labels, is_export_mode=False):
        """Calls the model_fn with required parameters."""
        self._validate_model_features_and_labels(features, labels, is_export_mode)
        model_fn_args = function_utils.fn_args(self._model_fn)
        kwargs = {}
        config = copy.deepcopy(self._config)
        params = copy.deepcopy(self._params)
        if 'labels' in model_fn_args:
            kwargs['labels'] = labels
        elif labels is not None:
            raise ValueError('model_fn does not take labels, but input_fn returns labels.')
        if 'mode' in model_fn_args:
            kwargs['mode'] = self._ctx.mode
        if 'config' in model_fn_args:
            kwargs['config'] = config
        if 'params' in model_fn_args:
            kwargs['params'] = params
        if 'params' not in model_fn_args:
            raise ValueError("model_fn ({}) does not include params argument, required by TPUEstimator to pass batch size as params['batch_size']".format(self._model_fn))
        if is_export_mode:
            batch_size_for_model_fn = None
        else:
            batch_size_for_model_fn = self._ctx.batch_size_for_model_fn
        if batch_size_for_model_fn is not None:
            _add_item_to_params(params, _BATCH_SIZE_KEY, batch_size_for_model_fn)
        running_on_cpu = self._ctx.is_running_on_cpu(is_export_mode)
        if not is_export_mode:
            _add_item_to_params(params, _USE_TPU_KEY, not running_on_cpu)
        if not running_on_cpu:
            user_context = tpu_context.TPUContext(internal_ctx=self._ctx, call_from_input_fn=False)
            _add_item_to_params(params, _CTX_KEY, user_context)
        estimator_spec = self._model_fn(features=features, **kwargs)
        if running_on_cpu and isinstance(estimator_spec, model_fn_lib._TPUEstimatorSpec):
            graph_context = tf.compat.v1.get_default_graph()._get_control_flow_context()
            try:
                if isinstance(graph_context, tpu._TPUInferenceContext):
                    tf.compat.v1.get_default_graph()._set_control_flow_context(graph_context.outer_context)
                return estimator_spec.as_estimator_spec()
            finally:
                tf.compat.v1.get_default_graph()._set_control_flow_context(graph_context)
        else:
            return estimator_spec

    def _verify_estimator_spec(self, estimator_spec):
        """Validates the estimator_spec."""
        if isinstance(estimator_spec, model_fn_lib._TPUEstimatorSpec):
            return estimator_spec
        err_msg = '{} returned by EstimatorSpec is not supported in TPUEstimator.'
        if estimator_spec.training_chief_hooks:
            raise ValueError(err_msg.format('training_chief_hooks') + 'If you want' + ' to pass training hooks, please pass via training_hooks.')
        if estimator_spec.scaffold:
            tf.compat.v1.logging.warn('EstimatorSpec.Scaffold is ignored by TPU train/eval. Please use TPUEstimatorSpec.')
        return estimator_spec