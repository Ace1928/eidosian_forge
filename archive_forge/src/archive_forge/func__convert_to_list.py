from typing import Any, Callable, Iterable, List, Optional, Union
from tensorflow.python.compiler.xla import xla
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import while_loop as while_loop_tf
from tensorflow.python.tpu import tensor_tracer
from tensorflow.python.tpu import tpu_feed
from tensorflow.python.tpu import tpu_function
from tensorflow.python.types import core as core_types
def _convert_to_list(xs):
    if not isinstance(xs, (list, tuple)):
        return [xs]
    else:
        return list(xs)