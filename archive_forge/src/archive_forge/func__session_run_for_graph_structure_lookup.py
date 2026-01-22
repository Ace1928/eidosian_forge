import collections
import functools
import glob
import os
import tempfile
import threading
import numpy as np
from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.core.util import event_pb2
from tensorflow.python.client import session
from tensorflow.python.debug.lib import debug_data
from tensorflow.python.debug.lib import debug_graphs
from tensorflow.python.debug.lib import debug_utils
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.lib.io import file_io
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import cond as tf_cond
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import parsing_ops
from tensorflow.python.ops import rnn
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_v1
from tensorflow.python.ops import variables
from tensorflow.python.ops import while_loop
import tensorflow.python.ops.tensor_array_grad  # pylint: disable=unused-import
from tensorflow.python.platform import googletest
from tensorflow.python.platform import test
from tensorflow.python.training import gradient_descent
def _session_run_for_graph_structure_lookup(self):
    with session.Session(config=no_rewrite_session_config()) as sess:
        u_name = 'testDumpGraphStructureLookup/u'
        v_name = 'testDumpGraphStructureLookup/v'
        w_name = 'testDumpGraphStructureLookup/w'
        u_init = constant_op.constant([2.0, 4.0])
        u = variable_v1.VariableV1(u_init, name=u_name)
        v = math_ops.add(u, u, name=v_name)
        w = math_ops.add(v, v, name=w_name)
        u.initializer.run()
        _, dump = self._debug_run_and_get_dump(sess, w, expected_partition_graph_count=self._expected_partition_graph_count)
    return (u_name, v_name, w_name, dump)