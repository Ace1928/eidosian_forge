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
def _testMakeInputFnIteratorWithDataset(self, distribution):
    dataset_fn = lambda: dataset_ops.Dataset.range(100)
    num_gpus = self._get_num_gpus()
    num_workers = 1
    expected_values = [[i + j for j in range(num_gpus)] * num_workers for i in range(0, 100, num_gpus)]
    with self.cached_session() as sess:
        input_fn = self._input_fn_to_test_input_context(dataset_fn, expected_num_replicas_in_sync=num_workers * num_gpus, expected_num_input_pipelines=num_workers, expected_input_pipeline_id=None)
        iterator = distribution.make_input_fn_iterator(input_fn)
        self._test_input_fn_iterator(iterator, distribution.extended.worker_devices, expected_values, sess)