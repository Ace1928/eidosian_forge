import collections
import copy
import os
import re
import shlex
from typing import List, Tuple
from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.python.client import session
from tensorflow.python.framework import convert_to_constants
from tensorflow.python.framework import ops as ops_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import versions
from tensorflow.python.grappler import tf_optimizer
from tensorflow.python.lib.io import file_io
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import sysconfig as sysconfig_lib
from tensorflow.python.platform import test
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import saver as saver_lib
Convert `signature_def` to tf2xla config.  Returns a `tf2xla.Config` proto.

  Args:
    signature_def: Instance of `SignatureDef`.
    variable_nodes_to_feed: List of tuples of form `(node_def, modified)`
      corresponding to VarHandleOp, and a boolean `modified` that describes
      whether the variable was modified during execution.

  Returns:
    An instance of `tf2xla.Config` proto.

  Raises:
    RuntimeError: If TensorFlow was not compiled with XLA.
  