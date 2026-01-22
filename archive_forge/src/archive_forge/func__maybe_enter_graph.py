import contextlib
from tensorflow.python.distribute import packed_distributed_variable as packed
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.tpu import tpu_replication
@contextlib.contextmanager
def _maybe_enter_graph(tensor):
    if context.executing_eagerly() or isinstance(tensor, ops.EagerTensor) or ops.has_default_graph():
        yield
    else:
        with tensor.graph.as_default():
            yield