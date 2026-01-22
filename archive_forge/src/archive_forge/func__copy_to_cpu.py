import atexit
import collections
import copy
import queue
import threading
import time
import weakref
from absl import logging
from tensorflow.python.checkpoint import checkpoint_context
from tensorflow.python.distribute import device_util
from tensorflow.python.distribute.sharded_variable import ShardedVariable
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.eager import executor
from tensorflow.python.framework import device as pydev
from tensorflow.python.framework import ops
from tensorflow.python.ops.resource_variable_ops import UninitializedVariable
from tensorflow.python.ops.variables import Variable
from tensorflow.python.saved_model.pywrap_saved_model import metrics
from tensorflow.python.util import object_identity
@def_function.function
def _copy_to_cpu(self):
    """Copy the checkpointed variables from the accelerator to the host CPU.

    TODO(chienchunh): Get the concrete function before firstly called to avoid
                      hangining the accelerators idle during function tracing.
    """
    for accelerator_var, cpu_var in self._object_map.items():
        if isinstance(accelerator_var, ShardedVariable) or hasattr(accelerator_var, _TPU_EMBEDDING_ATTR):
            continue
        with ops.device(cpu_var.device):
            cpu_var.assign(accelerator_var.read_value())
    for tpu_embedding in self._tpu_embedding_objects:
        tpu_embedding._retrieve_variables()