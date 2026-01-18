from tensorflow.python import tf2
from tensorflow.python.data.experimental.ops import counter
from tensorflow.python.data.experimental.ops import interleave_ops
from tensorflow.python.data.experimental.ops import random_ops
from tensorflow.python.data.experimental.ops import readers as exp_readers
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import readers
from tensorflow.python.eager import monitoring
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import control_flow_v2_toggles
from tensorflow.python.ops import variable_scope
from tensorflow.python.util.tf_export import tf_export
Disables TensorFlow 2.x behaviors.

  This function can be called at the beginning of the program (before `Tensors`,
  `Graphs` or other structures have been created, and before devices have been
  initialized. It switches all global behaviors that are different between
  TensorFlow 1.x and 2.x to behave as intended for 1.x.

  User can call this function to disable 2.x behavior during complex migrations.

  @compatibility(TF2)
  Using this function indicates that your software is not compatible
  with eager execution and `tf.function` in TF2.

  To migrate to TF2, rewrite your code to be compatible with eager execution.
  Please refer to the [migration guide]
  (https://www.tensorflow.org/guide/migrate) for additional resource on the
  topic.
  @end_compatibility
  