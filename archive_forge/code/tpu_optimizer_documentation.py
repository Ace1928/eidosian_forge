from tensorflow.python.framework import ops
from tensorflow.python.ops.losses import losses
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.tpu import tpu_function
from tensorflow.python.tpu.ops import tpu_ops
from tensorflow.python.training import optimizer
from tensorflow.python.util.tf_export import tf_export
Forwarding the variables from the underlying optimizer.