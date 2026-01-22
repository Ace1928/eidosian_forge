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
class _InputPipeline(object):
    """`_InputPipeline` handles invoking `input_fn` and piping to infeed queue.

  `_InputPipeline` abstracts the per-core/per-host `input_fn` invocation from
  call site.  To be precise, based on the configuration in
  `_InternalTPUContext`,  it invokes `input_fn` for all cores (usually
  multi-host TPU training) or for one host (usually for single-host TPU
  evaluation), and sends all `features` and `labels` returned by `input_fn` to
  TPU infeed. For per-core invocation, `features` and `labels` are piped to
  infeed directly, one tuple for each core. For per-host invocation,  `features`
  and `labels` are split at host (with respect to `batch_axis`) and piped to all
  cores accordingly.

  In addition, flatten/unflatten are handled by `_InputPipeline` also.  Model
  inputs returned by the `input_fn` can have one of the following forms:
  1. features
  2. (features, labels)
  3. ((arbitrarily nested structure of features), labels)

  Internally, form 1 is reformed to `(features, None)` as features and labels
  are passed separately to underlying methods. For TPU training, TPUEstimator
  may expect multiple `features` and `labels` tuples one for each core.

  TPUEstimator allows various different structures for inputs (namely `features`
  and `labels`).  Both `features` and `labels` can be any nested sturcture
  supported by TF nest (namely, dict, tuples, namedtuples or any nested
  structure of such of Tensors).  `labels` could be `None` as well.

  These are flattened before they are passed to the infeed/outfeed library
  as that expectes flattend lists.
  """

    class InputsStructureRecorder(object):
        """The recorder to record inputs structure."""

        def __init__(self, input_partition_dims=None):
            self._feature_structure = {}
            self._flattened_input_dims = None
            if input_partition_dims:
                assert len(input_partition_dims) <= 2, 'must have 1 or 2 elements.'
                if len(input_partition_dims) == 2:
                    self._feature_dims, self._label_dims = input_partition_dims
                else:
                    self._feature_dims = input_partition_dims[0]
                    self._label_dims = None
                assert self._feature_dims is not None, 'input_partition_dims[0] must not be None'
            else:
                self._feature_dims = None
                self._label_dims = None
            self._initialized = False

        @property
        def flattened_input_dims(self):
            assert self._initialized, 'InputsStructureRecorder is not initialized.'
            return self._flattened_input_dims

        def has_labels(self):
            return 'labels' in self._feature_structure

        def _flatten_input_dims(self, features, labels, feature_dims, label_dims):
            """Flatten input dims with the same order as flattened input tensors."""
            try:
                flattened_input_dims = data_nest.flatten_up_to(features, feature_dims)
            except TypeError as e:
                raise ValueError('TPUConfig.input_partition_dims[0] mismatched the structure of features. input_partition_dims[0]: {}, features {}. {}'.format(feature_dims, features, e))
            if labels is not None:
                if label_dims is not None:
                    try:
                        flattened_input_dims.extend(data_nest.flatten_up_to(labels, self._label_dims))
                    except TypeError as e:
                        raise ValueError('TPUConfig.input_partition_dims[1] mismatched the structure of labels. input_partition_dims[1]: {}, labels: {}. {}'.format(label_dims, labels, e))
                else:
                    num_label_tensors = len(data_nest.flatten(labels))
                    flattened_input_dims.extend([None] * num_label_tensors)
            return flattened_input_dims

        def validate_and_record_structure(self, features, labels):
            """Validates and records the structure of `features` and `labels`."""
            feature_names = _extract_key_names(features)
            label_names = _extract_key_names(labels)
            if not self._initialized:
                self._initialized = True
                if self._feature_dims is not None:
                    feature_dims_names = _extract_key_names(self._feature_dims)
                    if feature_dims_names != feature_names:
                        raise ValueError('TPUConfig.input_partition_dims[0] mismatched feature keys. Expected {}, got {}'.format(feature_names, feature_dims_names))
                    label_dims_names = _extract_key_names(self._label_dims)
                    if self._label_dims is not None and label_dims_names != label_names:
                        raise ValueError('TPUConfig.input_partition_dims[1] mismatched label keys. Expected {}, got {}'.format(label_names, label_dims_names))
                    self._flattened_input_dims = self._flatten_input_dims(features, labels, self._feature_dims, self._label_dims)

        def flatten_features_and_labels(self, features, labels, signals=None):
            """Flattens the `features` and `labels` to a single tensor list."""
            self.tensor_packer = TensorPacker(_TENSOR_PACKER_SMALL_FEATURE_DIM_SIZE, _TENSOR_PACKER_MINIMUM_NUM_SMALL_FEATURES_TO_GROUP)
            self.tensor_packer.maybe_concatenate_features(features)
            self._feature_structure['features'] = features
            if labels is not None:
                self._feature_structure['labels'] = labels
            if signals is not None:
                self._feature_structure['signals'] = signals
            return data_nest.flatten(self._feature_structure)

        def unflatten_features_and_labels(self, flattened_inputs):
            """Restores the flattened inputs to original features and labels form.

      Args:
        flattened_inputs: Flattened inputs for each shard.

      Returns:
        A tuple of (`features`, `labels`), where `labels` could be None.
        Each one, if present, should have identical structure (single tensor vs
        dict) as the one returned by input_fn.

      Raises:
        ValueError: If the number of expected tensors from `flattened_inputs`
          mismatches the recorded structure.
      """
            unflattened_inputs = data_nest.pack_sequence_as(self._feature_structure, flattened_inputs)
            features = unflattened_inputs['features']
            self.tensor_packer.maybe_split_features(features)
            return _Inputs(features, unflattened_inputs.get('labels'), signals=unflattened_inputs.get('signals'))

    def __init__(self, input_fn, batch_axis, ctx):
        """Constructor.

    Args:
      input_fn: input fn for train or eval.
      batch_axis: A python tuple of int values describing how each tensor
        produced by the Estimator `input_fn` should be split across the TPU
        compute shards.
      ctx: A `_InternalTPUContext` instance with mode.

    Raises:
      ValueError: If both `sharded_features` and `num_cores` are `None`.
    """
        self._inputs_structure_recorder = _InputPipeline.InputsStructureRecorder(ctx.input_partition_dims)
        self._sharded_per_core = ctx.is_input_sharded_per_core()
        self._input_fn = input_fn
        self._infeed_queue = None
        self._ctx = ctx
        self._batch_axis = batch_axis

    def generate_infeed_enqueue_ops_and_dequeue_fn(self):
        """Generates infeed enqueue ops and dequeue_fn."""
        enqueue_ops, all_hooks, run_infeed_loop_on_coordinator = self._invoke_input_fn_and_record_structure()
        self._validate_input_pipeline()

        def dequeue_fn():
            """dequeue_fn is used by TPU to retrieve the tensors."""
            values = self._infeed_queue.generate_dequeue_op(tpu_device=0)
            return self._inputs_structure_recorder.unflatten_features_and_labels(values)
        return (enqueue_ops, dequeue_fn, all_hooks, run_infeed_loop_on_coordinator)

    def _invoke_input_fn_and_record_structure(self):
        """Deploys the input pipeline and record input structure."""
        enqueue_ops = []
        infeed_queues = []
        all_dataset_initializers = []
        num_hosts = self._ctx.num_hosts
        tpu_host_placement_fn = self._ctx.tpu_host_placement_function
        run_infeed_loop_on_coordinator = True
        if self._sharded_per_core:
            for host_id in range(num_hosts):
                host_device = tpu_host_placement_fn(host_id=host_id)
                with tf.compat.v1.device(host_device):
                    with ops.name_scope('input_pipeline_task%d' % host_id):
                        enqueue_ops_fn, captured_infeed_queue = generate_per_core_enqueue_ops_fn_for_host(self._ctx, self._input_fn, self._inputs_structure_recorder, host_device, host_id)
                        if _WRAP_INPUT_FN_INTO_WHILE_LOOP:
                            run_infeed_loop_on_coordinator = False
                            enqueue_ops.append(_wrap_computation_in_while_loop(device=host_device, op_fn=enqueue_ops_fn))
                        else:
                            enqueue_ops.append(enqueue_ops_fn())
                        infeed_queues.append(captured_infeed_queue.get())
        elif self._ctx.is_input_broadcast_with_iterators():
            host_device = tpu_host_placement_fn(host_id=0)
            enqueue_ops_fn, captured_infeed_queue, dataset_initializer = generate_broadcast_enqueue_ops_fn(self._ctx, self._input_fn, self._inputs_structure_recorder, num_hosts)
            if dataset_initializer:
                all_dataset_initializers.append(dataset_initializer)
                run_infeed_loop_on_coordinator = False
                wrap_fn = _wrap_computation_in_while_loop if self._ctx.mode != model_fn_lib.ModeKeys.PREDICT else _wrap_computation_in_while_loop_with_stopping_signals
                enqueue_ops.append(wrap_fn(device=host_device, op_fn=enqueue_ops_fn))
            else:
                enqueue_ops.append(enqueue_ops_fn())
            infeed_queues.append(captured_infeed_queue.get())
        else:
            host_id_with_invocation_id_pair = []
            if not self._ctx.is_replica_across_hosts():
                for host_id in range(num_hosts):
                    invocation_index = host_id
                    host_id_with_invocation_id_pair.append((host_id, invocation_index))
            else:
                for replica_id in xrange(self._ctx.num_replicas):
                    invocation_index = replica_id
                    host_device, _ = self._ctx.device_for_replica(replica_id)
                    host_id = int(host_device.split('/task:')[1].split('/device:')[0])
                    host_id_with_invocation_id_pair.append((host_id, invocation_index))
            for host_id, invocation_index in host_id_with_invocation_id_pair:
                host_device = tpu_host_placement_fn(host_id=host_id)
                with tf.compat.v1.device(host_device):
                    with ops.name_scope('input_pipeline_task%d' % host_id):
                        if self._ctx.is_input_per_host_with_iterators():
                            enqueue_ops_fn, captured_infeed_queue, dataset_initializer = generate_per_host_v2_enqueue_ops_fn_for_host(self._ctx, self._input_fn, self._inputs_structure_recorder, host_device, host_id, invocation_index)
                        else:
                            enqueue_ops_fn, captured_infeed_queue, dataset_initializer = generate_per_host_enqueue_ops_fn_for_host(self._ctx, self._input_fn, self._inputs_structure_recorder, self._batch_axis, host_device, host_id)
                        if dataset_initializer:
                            all_dataset_initializers.append(dataset_initializer)
                            run_infeed_loop_on_coordinator = False
                            wrap_fn = _wrap_computation_in_while_loop if self._ctx.mode != model_fn_lib.ModeKeys.PREDICT else _wrap_computation_in_while_loop_with_stopping_signals
                            enqueue_ops.append(wrap_fn(device=host_device, op_fn=enqueue_ops_fn))
                        else:
                            enqueue_ops.append(enqueue_ops_fn())
                        infeed_queues.append(captured_infeed_queue.get())
        self._infeed_queue = infeed_queues[0]
        return (enqueue_ops, [util_lib.MultiHostDatasetInitializerHook(all_dataset_initializers)], run_infeed_loop_on_coordinator)

    def _validate_input_pipeline(self):
        """Validates the input pipeline.

    Perform some sanity checks to log user friendly information. We should
    error out to give users better error message. But, if
    _WRAP_INPUT_FN_INTO_WHILE_LOOP is False (legacy behavior), we cannot break
    user code, so, log a warning.

    Raises:
      RuntimeError: If the validation failed.
    """
        if tf.compat.v1.get_default_graph().get_collection(tf.compat.v1.GraphKeys.QUEUE_RUNNERS):
            err_msg = 'Input pipeline contains one or more QueueRunners. It could be slow and not scalable. Please consider converting your input pipeline to use `tf.data` instead (see https://www.tensorflow.org/guide/datasets for instructions.'
            if _WRAP_INPUT_FN_INTO_WHILE_LOOP:
                raise RuntimeError(err_msg)
            else:
                logging.warn(err_msg)