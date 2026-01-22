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
def _generate_dump_from_simple_addition_graph(self):
    with session.Session(config=no_rewrite_session_config()) as sess:
        u_init_val = np.array([[5.0, 3.0], [-1.0, 0.0]])
        v_init_val = np.array([[2.0], [-1.0]])
        u_name = 'u'
        v_name = 'v'
        w_name = 'w'
        u_init = constant_op.constant(u_init_val, shape=[2, 2])
        u = variable_v1.VariableV1(u_init, name=u_name)
        v_init = constant_op.constant(v_init_val, shape=[2, 1])
        v = variable_v1.VariableV1(v_init, name=v_name)
        w = math_ops.matmul(u, v, name=w_name)
        u.initializer.run()
        v.initializer.run()
        run_options = config_pb2.RunOptions(output_partition_graphs=True)
        debug_urls = 'file://%s' % self._dump_root
        debug_utils.add_debug_tensor_watch(run_options, '%s/read' % u_name, 0, debug_urls=debug_urls)
        debug_utils.add_debug_tensor_watch(run_options, '%s/read' % v_name, 0, debug_urls=debug_urls)
        run_metadata = config_pb2.RunMetadata()
        sess.run(w, options=run_options, run_metadata=run_metadata)
        self.assertEqual(self._expected_partition_graph_count, len(run_metadata.partition_graphs))
        dump = debug_data.DebugDumpDir(self._dump_root, partition_graphs=run_metadata.partition_graphs)
    simple_add_results = collections.namedtuple('SimpleAddResults', ['u_init_val', 'v_init_val', 'u', 'v', 'w', 'u_name', 'v_name', 'w_name', 'dump'])
    return simple_add_results(u_init_val, v_init_val, u, v, w, u_name, v_name, w_name, dump)