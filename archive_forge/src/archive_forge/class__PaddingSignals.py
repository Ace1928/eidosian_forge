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
class _PaddingSignals(object):
    """Signals class holding all logic to handle padding."""

    @staticmethod
    def pad_features_and_labels(features, labels, batch_size):
        """Pads out the batch dimension of features and labels."""
        real_batch_size = tf.compat.v1.shape(_PaddingSignals._find_any_tensor(features))[0]
        batch_size_tensor = tf.constant(batch_size, tf.dtypes.int32)
        check_greater = tf.compat.v1.debugging.assert_greater_equal(batch_size_tensor, real_batch_size, data=(batch_size_tensor, real_batch_size), message='The real batch size should not be greater than batch_size.')
        with tf.control_dependencies([check_greater]):
            missing_count = batch_size_tensor - real_batch_size

        def pad_single_tensor(tensor):
            """Pads out the batch dimension of a tensor to the complete batch_size."""
            rank = len(tensor.shape)
            assert rank > 0
            padding = tf.stack([[0, missing_count]] + [[0, 0]] * (rank - 1))
            padded_shape = (batch_size,) + tuple(tensor.shape[1:])
            padded_tensor = tf.compat.v1.pad(tensor, padding)
            padded_tensor.set_shape(padded_shape)
            return padded_tensor

        def nest_pad(tensor_or_dict):
            return tf.nest.map_structure(pad_single_tensor, tensor_or_dict)
        features = nest_pad(features)
        if labels is not None:
            labels = nest_pad(labels)
        padding_mask = _PaddingSignals._padding_mask(real_batch_size, missing_count, batch_size)
        return (padding_mask, features, labels)

    @staticmethod
    def slice_tensor_or_dict(tensor_or_dict, signals):
        """Slice the real Tensors according to padding mask in signals."""
        padding_mask = signals['padding_mask']
        batch_size = tf.compat.v1.shape(padding_mask)[0]

        def verify_batch_size(tensor):
            check_batch_size = tf.math.equal(batch_size, tensor.shape[0])
            with tf.control_dependencies([check_batch_size]):
                return tf.identity(tensor)

        def slice_single_tensor(tensor):
            rank = len(tensor.shape)
            assert rank > 0
            real_batch_size = batch_size - tf.math.reduce_sum(padding_mask)
            return verify_batch_size(tensor)[0:real_batch_size]
        sliced_padding_mask = slice_single_tensor(padding_mask)
        assert_padding_mask = tf.math.equal(tf.math.reduce_sum(sliced_padding_mask), 0)
        with tf.control_dependencies([assert_padding_mask]):
            should_stop = _StopSignals.should_stop(_StopSignals.as_scalar_stopping_signal(signals))
        is_full_batch = tf.math.equal(tf.math.reduce_sum(padding_mask), 0)

        def slice_fn(tensor):
            return tf.compat.v1.cond(tf.math.logical_or(should_stop, is_full_batch), lambda: verify_batch_size(tensor), lambda: slice_single_tensor(tensor))
        return tf.nest.map_structure(slice_fn, tensor_or_dict)

    @staticmethod
    def _find_any_tensor(batch_features):
        tensors = [x for x in tf.nest.flatten(batch_features) if isinstance(x, tf.Tensor)]
        if not tensors:
            raise ValueError('Cannot find any Tensor in features dict.')
        return tensors[0]

    @staticmethod
    def _padding_mask(real_batch_size, missing_count, batch_size):
        padding_mask = tf.concat([tf.zeros((real_batch_size,), dtype=tf.dtypes.int32), tf.ones((missing_count,), dtype=tf.dtypes.int32)], axis=0)
        padding_mask.set_shape((batch_size,))
        return padding_mask