from typing import Any, Callable, List, Optional, Text, Tuple, Union
from absl import logging
from tensorflow.core.framework import attr_value_pb2
from tensorflow.python.distribute import device_util
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.framework import device as pydev
from tensorflow.python.framework import errors
from tensorflow.python.framework import func_graph
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import variables
from tensorflow.python.tpu import device_assignment as device_assignment_lib
from tensorflow.python.tpu.ops import tpu_ops
from tensorflow.python.types import core as core_types
from tensorflow.python.util import compat
from tensorflow.python.util.tf_export import tf_export
def _set_outside_compilation_attributes(self, op: ops.Operation) -> None:
    op._set_attr(_OUTSIDE_COMPILATION_ATTR, attr_value_pb2.AttrValue(s=compat.as_bytes(self._name)))
    if self._is_map_outside_compilation:
        op._set_attr(_MAP_OUTSIDE_COMPILATION_ATTR, attr_value_pb2.AttrValue(b=True))