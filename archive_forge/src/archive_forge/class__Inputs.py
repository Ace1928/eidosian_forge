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
class _Inputs(object):
    """A data structure representing the input_fn returned values.

  This also supports the returned value from input_fn as `Dataset`.
  """

    def __init__(self, features=None, labels=None, dataset=None, signals=None):
        if dataset is not None and (features is not None or labels is not None or signals is not None):
            raise RuntimeError('Internal Error: Either (features and labels) or dataset should be provided, not both. Please file bug')
        self._features = features
        self._labels = labels
        self._signals = signals
        self._dataset = dataset
        self._iterator = None

    @staticmethod
    def from_input_fn(return_values):
        """Returns an `_Inputs` instance according to `input_fn` return value."""
        if isinstance(return_values, tf.compat.v2.data.Dataset):
            dataset = return_values
            return _Inputs(dataset=dataset)
        features, labels = _Inputs._parse_inputs(return_values)
        return _Inputs(features, labels)

    @staticmethod
    def _parse_inputs(return_values):
        if isinstance(return_values, tuple):
            features, labels = return_values
        else:
            features, labels = (return_values, None)
        return (features, labels)

    @property
    def is_dataset(self):
        """Returns True if the return value from input_fn is Dataset."""
        return self._dataset is not None

    def dataset_initializer(self):
        """Returns the dataset's initializer.

    The initializer must be run before calling `features_and_labels`.
    """
        self._iterator = tf.compat.v1.data.make_initializable_iterator(self._dataset)
        return self._iterator.initializer

    def features_and_labels(self):
        """Gets `features` and `labels`."""
        if self.is_dataset:
            if self._iterator is None:
                raise RuntimeError('Internal error: Must run dataset_initializer before calling features_and_labels(). Please file a bug!')
            return _Inputs._parse_inputs(self._iterator.get_next())
        return (self._features, self._labels)

    def signals(self):
        return self._signals

    @property
    def dataset(self):
        return self._dataset