import abc
import functools
import queue
import threading
import warnings
import numpy as np
from tensorflow.core.framework import dataset_metadata_pb2
from tensorflow.core.framework import dataset_options_pb2
from tensorflow.core.framework import graph_pb2
from tensorflow.core.protobuf import struct_pb2
from tensorflow.python import tf2
from tensorflow.python.data.ops import dataset_autograph
from tensorflow.python.data.ops import debug_mode
from tensorflow.python.data.ops import iterator_ops
from tensorflow.python.data.ops import options as options_lib
from tensorflow.python.data.ops import structured_function
from tensorflow.python.data.util import nest
from tensorflow.python.data.util import structure
from tensorflow.python.data.util import traverse
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.eager import wrap_function
from tensorflow.python.framework import auto_control_deps
from tensorflow.python.framework import auto_control_deps_utils as acd_utils
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import function
from tensorflow.python.framework import ops
from tensorflow.python.framework import random_seed as core_random_seed
from tensorflow.python.framework import sparse_tensor as sparse_tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import type_spec
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import cond
from tensorflow.python.ops import control_flow_assert
from tensorflow.python.ops import gen_dataset_ops
from tensorflow.python.ops import gen_io_ops
from tensorflow.python.ops import gen_parsing_ops
from tensorflow.python.ops import logging_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import string_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.saved_model import nested_structure_coder
from tensorflow.python.trackable import asset
from tensorflow.python.trackable import base as tracking_base
from tensorflow.python.trackable import resource as resource_lib
from tensorflow.python.types import data as data_types
from tensorflow.python.types import trace
from tensorflow.python.util import deprecation
from tensorflow.python.util import lazy_loader
from tensorflow.python.util import nest as tf_nest
from tensorflow.python.util.compat import collections_abc
from tensorflow.python.util.tf_export import tf_export
def _make_one_shot_iterator(self):
    if context.executing_eagerly():
        with ops.colocate_with(self._variant_tensor):
            return iterator_ops.OwnedIterator(self)
    _ensure_same_dataset_graph(self)
    allowlisted_stateful_ops = traverse.obtain_capture_by_value_ops(self)
    graph_level_seed, op_level_seed = core_random_seed.get_seed(None)

    @function.Defun(capture_by_value=True, allowlisted_stateful_ops=allowlisted_stateful_ops)
    def _make_dataset():
        """Factory function for a dataset."""
        if graph_level_seed is not None:
            assert op_level_seed is not None
            core_random_seed.set_random_seed((graph_level_seed + 89284321 * op_level_seed) % (2 ** 63 - 1))
        dataset = self._apply_debug_options()
        return dataset._variant_tensor
    try:
        _make_dataset.add_to_graph(ops.get_default_graph())
    except ValueError as err:
        if 'Cannot capture a stateful node' in str(err):
            raise ValueError('{}: A likely cause of this error is that the dataset for which you are calling `make_one_shot_iterator()` captures a stateful object, such as a `tf.Variable` or `tf.lookup.StaticHashTable`, which is not supported. Use `make_initializable_iterator()` instead.'.format(err)) from None
        else:
            raise
    with ops.colocate_with(self._variant_tensor):
        return iterator_ops.Iterator(gen_dataset_ops.one_shot_iterator(dataset_factory=_make_dataset, **self._flat_structure), None, get_legacy_output_types(self), get_legacy_output_shapes(self), get_legacy_output_classes(self))