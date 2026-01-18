import threading
from tensorflow.core.framework import attr_value_pb2
from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.ops import gen_resource_variable_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.trackable import base as trackable
from tensorflow.python.util import compat
from tensorflow.python.util import tf_contextlib
Lazily initialized variables.

    The major use case for this class is to serve as a memory efficient
    alternative for tf.Variable. The resource handle of this class is point to
    nothing, which mean it will raise error when its value is fetched in a eager
    context. Having said that, it will perform like a normal tf.Variable when
    using with graph tensor, like KerasTensor produced from tf.keras.Input.
    