import collections
import enum
from typing import Any, Callable, Iterable, List, Optional, Text, Tuple, Union
from absl import logging
import numpy as np
from tensorflow.compiler.tf2xla.python import xla as tf2xla
from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.protobuf.tpu import dynamic_padding_pb2 as dynamic_padding
from tensorflow.core.protobuf.tpu import tpu_embedding_configuration_pb2 as embedding_pb2
from tensorflow.python import tf2
from tensorflow.python.compiler.xla import xla
from tensorflow.python.framework import auto_control_deps
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import config
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import func_graph
from tensorflow.python.framework import function
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import cond
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.tpu import device_assignment as device_assignment_lib
from tensorflow.python.tpu import tensor_tracer
from tensorflow.python.tpu import tpu_feed
from tensorflow.python.tpu import tpu_function
from tensorflow.python.tpu import tpu_name_util
from tensorflow.python.tpu import tpu_replication
from tensorflow.python.tpu.ops import tpu_ops
from tensorflow.python.types import core as core_types
from tensorflow.python.util import compat
from tensorflow.python.util import nest
from tensorflow.python.util import object_identity
from tensorflow.python.util import traceback_utils
from tensorflow.python.util import variable_utils
from tensorflow.python.util.tf_export import tf_export
def validate_inference_rewrite_for_variables(graph: ops.Graph):
    """Validates whether rewrite_for_inference() 'worked' for variables.

     The rewrite_for_inference() method is supposed to append GuaranteeConstOps
     after ReadVariableOps, but this mechanism works only if you are using
     tf.compat.v1.get_variable() to create and access variables in your tpu
     computation. This validation method can be called immediately after calling
     tpu.rewrite_for_inference() to check whether GuaranteeConstOps where added
     to the graph.

     Typical usages:
       tpu.validate_inference_rewrite_for_variables(
           tf.compat.v1.get_default_graph())

       tpu.validate_inference_rewrite_for_variables(sess.graph)

  Args:
    graph: The graph which needs to be validated.
  Raises:
    RuntimeError: if validation failed.
  """
    if not any((x.type == 'GuaranteeConst' for x in graph.get_operations())):
        raise RuntimeError('No GuaranteeConst ops found in the graph after running tpu.rewrite_for_inference(...). Please check that you are using tf.get_variable() to create and access variables in your tpu computation.')