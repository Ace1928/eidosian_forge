from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from collections import abc
import contextlib
from tensorflow.python.compiler.xla.experimental import xla_sharding
from tensorflow.python.distribute import tpu_util
from tensorflow.python.eager import context
from tensorflow.python.framework import config
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_conversion_registry
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_resource_variable_ops
from tensorflow.python.ops import gen_tpu_partition_ops as tpu_partition_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables as variables_lib
from tensorflow.python.saved_model import save_context
from tensorflow.python.trackable import base as trackable
def _gather_saveables_for_saved_model(self):
    return {trackable.VARIABLE_VALUE_KEY: self._vars[0]}