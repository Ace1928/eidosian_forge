import functools
import os
import tempfile
import numpy as np
from tensorflow.core.protobuf import config_pb2
from tensorflow.core.util import event_pb2
from tensorflow.python.client import session as session_lib
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.distribute import collective_all_reduce_strategy as mwms_lib
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import distribute_utils
from tensorflow.python.distribute import mirrored_strategy as mirrored_lib
from tensorflow.python.distribute import reduce_util
from tensorflow.python.distribute import tpu_strategy
from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.eager import test
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.lib.io import tf_record
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import init_ops_v2
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import summary_ops_v2 as summary_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variable_v1
from tensorflow.python.ops import variables
from tensorflow.python.platform import gfile
from tensorflow.python.training import optimizer
from tensorflow.python.training import training_util
from tensorflow.python.util import nest
from tensorflow.python.util import tf_inspect
def _test_input_fn_iterable(self, strategy, input_fn, expected_values, ignore_order=False):
    assert_same = self.assertCountEqual if ignore_order else self.assertEqual
    iterable = strategy.distribute_datasets_from_function(input_fn)
    if context.executing_eagerly():
        iterator = iter(iterable)
        for expected_value in expected_values:
            computed_value = self.evaluate(list(strategy.experimental_local_results(next(iterator))))
            assert_same(expected_value, computed_value)
        with self.assertRaises(StopIteration):
            self.evaluate(strategy.experimental_local_results(next(iterator)))
        iterator = iter(iterable)
        for expected_value in expected_values:
            computed_value = self.evaluate(list(strategy.experimental_local_results(next(iterator))))
            assert_same(expected_value, computed_value)
    else:
        iterator = dataset_ops.make_initializable_iterator(iterable)
        self._test_input_fn_iterator(iterator, strategy.extended.worker_devices, expected_values, test_reinitialize=True, ignore_order=ignore_order)