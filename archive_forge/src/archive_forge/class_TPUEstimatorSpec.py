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
@estimator_export(v1=['estimator.tpu.TPUEstimatorSpec'])
class TPUEstimatorSpec(model_fn_lib._TPUEstimatorSpec):
    """Ops and objects returned from a `model_fn` and passed to `TPUEstimator`.

  See `EstimatorSpec` for `mode`, `predictions`, `loss`, `train_op`, and
  `export_outputs`.

  For evaluation, `eval_metrics `is a tuple of `metric_fn` and `tensors`, where
  `metric_fn` runs on CPU to generate metrics and `tensors` represents the
  `Tensor`s transferred from TPU system to CPU host and passed to `metric_fn`.
  To be precise, TPU evaluation expects a slightly different signature from the
  `tf.estimator.Estimator`. While `EstimatorSpec.eval_metric_ops` expects a
  dict, `TPUEstimatorSpec.eval_metrics` is a tuple of `metric_fn` and `tensors`.
  The `tensors` could be a list of `Tensor`s or dict of names to `Tensor`s. The
  `tensors` usually specify the model logits, which are transferred back from
  TPU system to CPU host. All tensors must have be batch-major, i.e., the batch
  size is the first dimension. Once all tensors are available at CPU host from
  all shards, they are concatenated (on CPU) and passed as positional arguments
  to the `metric_fn` if `tensors` is list or keyword arguments if `tensors` is
  a dict. `metric_fn` takes the `tensors` and returns a dict from metric string
  name to the result of calling a metric function, namely a `(metric_tensor,
  update_op)` tuple. See `TPUEstimator` for MNIST example how to specify the
  `eval_metrics`.

  `scaffold_fn` is a function running on CPU to generate the `Scaffold`. This
  function should not capture any Tensors in `model_fn`.

  `host_call` is a tuple of a `function` and a list or dictionary of `tensors`
  to pass to that function and returns a list of Tensors. `host_call` currently
  works for train() and evaluate(). The Tensors returned by the function is
  executed on the CPU on every step, so there is communication overhead when
  sending tensors from TPU to CPU. To reduce the overhead, try reducing the
  size of the tensors. The `tensors` are concatenated along their major (batch)
  dimension, and so must be >= rank 1. The `host_call` is useful for writing
  summaries with `tf.summary.create_file_writer`.

  @compatibility(TF2)
  TPU Estimator manages its own TensorFlow graph and session, so it is not
  compatible with TF2 behaviors. We recommend that you migrate to the newer
  `tf.distribute.TPUStrategy`. See the
  [TPU guide](https://www.tensorflow.org/guide/tpu) for details.
  @end_compatibility
  """

    def __new__(cls, mode, predictions=None, loss=None, train_op=None, eval_metrics=None, export_outputs=None, scaffold_fn=None, host_call=None, training_hooks=None, evaluation_hooks=None, prediction_hooks=None):
        """Creates a validated `TPUEstimatorSpec` instance."""
        cls._host_calls = {}
        if eval_metrics is not None:
            cls._host_calls['eval_metrics'] = eval_metrics
        if host_call is not None:
            cls._host_calls['host_call'] = host_call
        _OutfeedHostCall.validate(cls._host_calls)
        training_hooks = tuple(training_hooks or [])
        evaluation_hooks = tuple(evaluation_hooks or [])
        prediction_hooks = tuple(prediction_hooks or [])
        for hook in training_hooks + evaluation_hooks + prediction_hooks:
            if not isinstance(hook, tf.compat.v1.train.SessionRunHook):
                raise TypeError('All hooks must be SessionRunHook instances, given: {}'.format(hook))
        return super(TPUEstimatorSpec, cls).__new__(cls, mode=mode, predictions=predictions, loss=loss, train_op=train_op, eval_metrics=eval_metrics, export_outputs=export_outputs, scaffold_fn=scaffold_fn, host_call=host_call, training_hooks=training_hooks, evaluation_hooks=evaluation_hooks, prediction_hooks=prediction_hooks)

    def as_estimator_spec(self):
        """Creates an equivalent `EstimatorSpec` used by CPU train/eval."""
        host_call_ret = _OutfeedHostCall.create_cpu_hostcall(self._host_calls)
        eval_metric_ops = None
        if self.eval_metrics is not None:
            eval_metric_ops = host_call_ret['eval_metrics']
        hooks = None
        if self.host_call is not None:
            hooks = [_OutfeedHostCallHook(host_call_ret['host_call'])]
        loss = self.loss
        if tensor_tracer.TensorTracer.is_enabled() and self.train_op is not None:
            tt = tensor_tracer.TensorTracer()
            loss = tt.trace_cpu(tf.compat.v1.get_default_graph(), loss, self.train_op)
        hooks = tuple(hooks or [])
        scaffold = self.scaffold_fn() if self.scaffold_fn else None
        return model_fn_lib.EstimatorSpec(mode=self.mode, predictions=self.predictions, loss=loss, train_op=self.train_op, eval_metric_ops=eval_metric_ops, export_outputs=self.export_outputs, scaffold=scaffold, training_hooks=self.training_hooks + hooks, evaluation_hooks=self.evaluation_hooks + hooks, prediction_hooks=self.prediction_hooks + hooks)