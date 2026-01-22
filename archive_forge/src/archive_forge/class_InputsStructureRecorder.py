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